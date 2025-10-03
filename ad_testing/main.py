import urllib.request
import os
import event_stream
import numpy
from tonic import transforms
import matplotlib.pyplot as plt
import matplotlib.colors
import PIL.Image
import PIL.ImageFilter
import skimage.measure
import requests
import json
import time
from dotenv import load_dotenv


def download_dataset():
    """Download the star dataset from GitHub to the dataset folder."""
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    stars_pattern = "stars_cut.es"
    stars_path = os.path.join(dataset_dir, stars_pattern)

    print("Checking for star dataset...")
    if not os.path.exists(stars_path):
        print("Downloading star dataset...")
        url = f"https://github.com/neuromorphicsystems/tutorials/raw/main/data/{stars_pattern}"
        stars_path, _ = urllib.request.urlretrieve(url, stars_path)
        print(f"Downloaded dataset to: {stars_path}")
    else:
        print(f"Dataset already exists at: {stars_path}")

    return stars_path, dataset_dir


def load_event_stream(stars_path):
    """Load event-based data from the .es file."""
    print("\nLoading event data...")
    with event_stream.Decoder(stars_path) as decoder:
        width = decoder.width
        height = decoder.height
        events = numpy.concatenate([chunk for chunk in decoder])
        print(f"Sensor resolution: {width} x {height}")
        print(f"Total events loaded: {len(events)}")

    # Convert events to format expected by Tonic
    events = events.view(dtype=events.dtype.descr[:3] + [(('on', 'p'), '?')])

    return events, width, height


def save_basic_star_map(events, width, height, dataset_dir):
    """Save basic accumulated star map before any corrections."""
    print("\nGenerating basic star map (raw event accumulation)...")

    image_transform = transforms.ToImage(sensor_size=(width, height, 2))
    accumulated_frame = image_transform(events)
    accumulated_frame = numpy.flip(accumulated_frame.sum(0), axis=0)

    plt.figure(figsize=(12, 8))
    plt.imshow(
        accumulated_frame,
        norm=matplotlib.colors.LogNorm(vmax=numpy.percentile(accumulated_frame, 99.9)),
        cmap="magma"
    )
    plt.title("Raw Star Map - Before Corrections")
    plt.colorbar(label="Event Count (log scale)")

    output_path = os.path.join(dataset_dir, 'star_map_basic_raw.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Basic star map saved to: {output_path}")
    plt.close()

    return output_path


def filter_noise(events, width, height, time_window=10000):
    """Apply neighbour support noise filtering to remove spurious events."""
    print(f"\nApplying noise filtering (time window: {time_window} µs)...")
    timestamps = numpy.zeros((width, height))
    filtered_events = []

    for t, x, y, on in events:
        timestamps[x][y] = t
        # Check if any neighbouring pixel had an event within the time window
        if (
            (x > 0 and timestamps[x - 1][y] + time_window > t)
            or (x < width - 1 and timestamps[x + 1][y] + time_window > t)
            or (y > 0 and timestamps[x][y - 1] + time_window > t)
            or (y < height - 1 and timestamps[x][y + 1] + time_window > t)
        ):
            filtered_events.append((t, x, y, on))

    filtered_events = numpy.array(filtered_events, dtype=events.dtype)
    reduction = ((len(events) - len(filtered_events)) / len(events)) * 100
    print(f"Original events: {len(events)}")
    print(f"Filtered events: {len(filtered_events)}")
    print(f"Reduced by: {reduction:.2f}%")

    return filtered_events


def warp_events(events, width, height, vx=-1.861, vy=0.549):
    """Warp events to compensate for sidereal motion."""
    print(f"\nWarping events (vx={vx} px/s, vy={vy} px/s)...")

    t0 = events["t"][0]
    duration = events["t"][-1] - t0

    warped_events = numpy.zeros(len(events), dtype=events.dtype)

    # Warp the events (shear operation)
    warped_events["t"] = events["t"]
    if vx > 0:
        warped_events["x"] = events["x"] + numpy.floor(
            (vx / 1e6) * (duration - (events["t"] - t0))
        ).astype(numpy.int16)
    else:
        warped_events["x"] = events["x"] + numpy.floor(
            (-vx / 1e6) * (events["t"] - t0)
        ).astype(numpy.int16)
    if vy > 0:
        warped_events["y"] = events["y"] + numpy.floor(
            (vy / 1e6) * (duration - (events["t"] - t0))
        ).astype(numpy.int16)
    else:
        warped_events["y"] = events["y"] + numpy.floor(
            (-vy / 1e6) * (events["t"] - t0)
        ).astype(numpy.int16)
    warped_events["on"] = events["on"]

    # Calculate new dimensions
    new_height = height + int(numpy.floor(abs(vy / 1e6) * duration))
    new_width = width + int(numpy.floor(abs(vx / 1e6) * duration))

    print(f"Warped frame size: {new_width} x {new_height}")

    return warped_events, new_width, new_height


def accumulate_and_filter_frame(warped_events, width, height):
    """Accumulate warped events into a frame and apply median filter."""
    print("\nAccumulating warped events into frame...")

    # Accumulate the warped events
    warped_frame = numpy.ones((height, width))
    numpy.add.at(
        warped_frame,
        (height - 1 - warped_events["y"], warped_events["x"]),
        1,
    )

    # Apply median filter to remove noise
    print("Applying median filter...")
    image = PIL.Image.fromarray(warped_frame)
    filtered_image = image.filter(PIL.ImageFilter.MedianFilter(5))
    filtered_frame = numpy.array(filtered_image)

    return filtered_frame


def detect_and_label_stars(filtered_frame, percentile_threshold=99.8):
    """
    Detect and label individual stars in the frame.

    Args:
        filtered_frame: The accumulated and filtered star image
        percentile_threshold: Percentile threshold for star detection (higher = fewer stars)
                            Default 99.8 means only brightest 0.2% of pixels
                            Try 99.5 for more stars, 99.9 for fewer/brighter stars only
    """
    print(f"\nDetecting and labeling stars (threshold: {percentile_threshold}%)...")

    # Create binary mask using percentile threshold
    nonzero = filtered_frame[filtered_frame > 0.0]
    threshold = numpy.percentile(nonzero, percentile_threshold)
    binary_mask = filtered_frame > threshold

    print(f"Threshold value: {threshold:.2f}")
    print(f"Pixels above threshold: {numpy.sum(binary_mask)}")

    # Label connected regions
    labels_array, num_stars = skimage.measure.label(
        binary_mask, connectivity=1, background=0, return_num=True
    )

    print(f"Detected {num_stars} stars")

    return labels_array, num_stars, binary_mask


def calculate_star_centers(warped_events, labels_array, num_stars, warped_height):
    """Calculate the center position of each detected star."""
    print("\nCalculating star centers...")

    # Label the warped events
    labelled_events = numpy.zeros(
        len(warped_events),
        dtype=[
            ("t", "<u8"),
            ("x", "<u2"),
            ("y", "<u2"),
            ("on", "?"),
            ("label", "<u4"),
        ],
    )

    labelled_events["t"] = warped_events["t"]
    labelled_events["x"] = warped_events["x"]
    labelled_events["y"] = warped_events["y"]
    labelled_events["on"] = warped_events["on"]
    labelled_events["label"] = labels_array[
        warped_height - 1 - warped_events["y"], warped_events["x"]
    ]

    # Calculate center of mass for each star
    centers = numpy.zeros((num_stars + 1, 2))
    for label in range(1, num_stars + 1):
        mask = labelled_events["label"] == label
        centers[label] = (
            numpy.mean(labelled_events["x"][mask]),
            numpy.mean(labelled_events["y"][mask]),
        )

    return centers, labelled_events


def perform_astrometry_via_api(dataset_dir):
    """
    Perform astrometry using Astrometry.net API programmatically.
    No manual upload needed - all done in code.

    Args:
        dataset_dir: Directory containing the star map image
    """
    print("\n" + "="*60)
    print("PERFORMING ASTROMETRY VIA API")
    print("="*60)

    api_result = submit_to_astrometry_api(dataset_dir)

    if api_result is not None:
        print("\n" + "="*60)
        print("ASTROMETRY SOLUTION FOUND!")
        print("="*60)
        print(f"Center RA:  {api_result['ra']:.4f} degrees")
        print(f"Center DEC: {api_result['dec']:.4f} degrees")
        print(f"Scale: {api_result['scale']:.4f} arcsec/pixel")
        print(f"Orientation: {api_result['orientation']:.2f} degrees")
        print("="*60)
        return api_result
    else:
        print("\n" + "="*60)
        print("ASTROMETRY FAILED")
        print("="*60)
        print("Could not solve via API.")
        print("Please check your API key or upload manually to:")
        print("https://nova.astrometry.net/upload")
        print("="*60)
        return None


def submit_to_astrometry_api(dataset_dir):
    """
    Submit image to Astrometry.net API programmatically.
    Requires API key in astrometry_api_key.env file.
    """
    print("Checking for Astrometry.net API key...")

    # Load environment variables from .env file
    env_file = 'ad_testing\\astrometry_api_key.env'
    load_dotenv(env_file)

    api_key = os.getenv('ASTROMETRY_API_KEY')

    if not api_key:
        print("\nNo API key found.")
        print("To enable automatic astrometry:")
        print(f"1. Get free API key from: https://nova.astrometry.net/api_help")
        print(f"2. Create '{env_file}' file with:")
        print(f"   astrometry_api_key=YOUR_API_KEY_HERE")
        return None

    print("API key found. Submitting to Astrometry.net...")

    try:
        image_path = os.path.join(dataset_dir, 'star_map_for_api.png')

        # Login
        login_url = 'http://nova.astrometry.net/api/login'
        login_data = {'request-json': json.dumps({'apikey': api_key})}
        response = requests.post(login_url, data=login_data)

        if response.status_code != 200 or response.json()['status'] != 'success':
            print("API login failed")
            return None

        session_key = response.json()['session']
        print(f"Logged in successfully")

        # Upload image
        upload_url = 'http://nova.astrometry.net/api/upload'
        upload_data = {
            'request-json': json.dumps({
                'session': session_key,
                'allow_modifications': 'd',
                'publicly_visible': 'n',
                'allow_commercial_use': 'n',
            })
        }

        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(upload_url, data=upload_data, files=files)

        if response.status_code != 200 or response.json()['status'] != 'success':
            print("Upload failed")
            return None

        subid = response.json()['subid']
        print(f"Upload successful. Submission ID: {subid}")
        print("Waiting for solution (1-5 minutes)...")

        # Wait initial time before polling
        time.sleep(5)

        # Poll for results
        max_attempts = 60
        for attempt in range(max_attempts):
            status_url = f'http://nova.astrometry.net/api/submissions/{subid}'

            try:
                response = requests.get(status_url)
            except requests.RequestException:
                # Network error, continue waiting
                time.sleep(5)
                continue

            if response.status_code == 200 and response.text.strip():
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    # Response not ready yet, continue waiting
                    print("Waiting for response...")
                    time.sleep(5)
                    continue

                job_ids = result.get('jobs', [])

                if job_ids:
                    job_id = job_ids[0]
                    job_url = f'http://nova.astrometry.net/api/jobs/{job_id}/info'

                    try:
                        job_response = requests.get(job_url)
                    except requests.RequestException:
                        time.sleep(5)
                        continue

                    if job_response.status_code == 200:
                        try:
                            job_info = job_response.json()
                        except json.JSONDecodeError:
                            time.sleep(5)
                            continue

                        if job_info['status'] == 'success':
                            calib = job_info.get('calibration', {})
                            print("\nSolution found!")
                            return {
                                'ra': calib.get('ra', 0),
                                'dec': calib.get('dec', 0),
                                'scale': calib.get('pixscale', 0),
                                'orientation': calib.get('orientation', 0)
                            }
                        elif job_info['status'] == 'failure':
                            print("\nJob failed to solve")
                            return None

            time.sleep(5)
            if attempt % 6 == 0 and attempt > 0:
                print(f"Still waiting... ({attempt*5}s)")

        print("\nTimeout waiting for solution")
        return None

    except Exception as e:
        print(f"\nAPI error: {e}")
        return None


def visualize_detected_stars(filtered_frame, centers, num_stars, dataset_dir):
    """Visualize detected stars without astrometry solution."""
    print("\nVisualizing detected stars...")

    # Create figure with no axes
    fig = plt.figure(figsize=(filtered_frame.shape[1]/100, filtered_frame.shape[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Plot star map
    ax.imshow(filtered_frame, norm=matplotlib.colors.LogNorm(), cmap="magma")

    # Plot detected star centers
    ax.scatter(
        x=centers[1:, 0],
        y=filtered_frame.shape[0] - 1 - centers[1:, 1],
        marker="o",
        color="#00ff00",
        s=150
    )

    output_path = os.path.join(dataset_dir, 'star_map_detected.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    print(f"Star map saved to: {output_path}")
    plt.close()

    return output_path


def save_binary_image_for_api(filtered_frame, dataset_dir):
    """Save a clean binary accumulated image for Astrometry.net API submission."""
    print("\nSaving binary image for API submission...")

    # Create clean image without any overlays
    fig = plt.figure(figsize=(filtered_frame.shape[1]/100, filtered_frame.shape[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Plot just the accumulated frame
    ax.imshow(filtered_frame, norm=matplotlib.colors.LogNorm(), cmap="gray")

    output_path = os.path.join(dataset_dir, 'star_map_for_api.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    print(f"API image saved to: {output_path}")
    plt.close()

    return output_path


def save_detected_stars_data(centers, num_stars, dataset_dir):
    """Save detected star positions to a text file."""
    output_file = os.path.join(dataset_dir, 'detected_stars_info.txt')

    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DETECTED STARS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total detected stars: {num_stars}\n\n")

        f.write("="*60 + "\n")
        f.write("STAR CENTERS (Pixel Coordinates)\n")
        f.write("="*60 + "\n")
        f.write("Star ID | X (px) | Y (px)\n")
        f.write("-"*60 + "\n")
        for i, center in enumerate(centers[1:], 1):
            f.write(f"{i:7d} | {center[0]:7.2f} | {center[1]:7.2f}\n")

    print(f"Star data saved to: {output_file}")
    return output_file


def save_astrometry_results(astrometry_result, centers, num_stars, dataset_dir):
    """Save astrometry solution to a text file."""
    output_file = os.path.join(dataset_dir, 'astrometry_solution.txt')

    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ASTROMETRY SOLUTION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Center Right Ascension (RA):  {astrometry_result['ra']:.6f} degrees\n")
        f.write(f"Center Declination (DEC):     {astrometry_result['dec']:.6f} degrees\n")
        f.write(f"Scale:                        {astrometry_result['scale']:.6f} arcsec/pixel\n")
        f.write(f"Orientation:                  {astrometry_result['orientation']:.6f} degrees\n")
        f.write(f"Detected stars:               {num_stars}\n\n")

        f.write("="*60 + "\n")
        f.write("DETECTED STAR CENTERS (Pixel Coordinates)\n")
        f.write("="*60 + "\n")
        f.write("Star ID | X (px) | Y (px)\n")
        f.write("-"*60 + "\n")
        for i, center in enumerate(centers[1:], 1):
            f.write(f"{i:7d} | {center[0]:7.2f} | {center[1]:7.2f}\n")

    print(f"Astrometry solution saved to: {output_file}")
    return output_file


def generate_html_visualization(astrometry_result, centers, num_stars, frame_shape, dataset_dir):
    """Generate interactive HTML visualization of the astrometry solution."""
    print("\nGenerating interactive HTML visualization...")

    # Read template file with UTF-8 encoding
    template_path = os.path.join(os.path.dirname(__file__), 'visualization_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()

    # Convert RA/DEC to HMS/DMS format
    ra_deg = astrometry_result['ra']
    dec_deg = astrometry_result['dec']

    ra_hours = int(ra_deg / 15)
    ra_minutes = int((ra_deg / 15 - ra_hours) * 60)
    ra_seconds = ((ra_deg / 15 - ra_hours) * 60 - ra_minutes) * 60

    dec_sign = '-' if dec_deg < 0 else '+'
    dec_abs = abs(dec_deg)
    dec_degrees = int(dec_abs)
    dec_arcmin = int((dec_abs - dec_degrees) * 60)
    dec_arcsec = ((dec_abs - dec_degrees) * 60 - dec_arcmin) * 60

    # Calculate field of view
    fov_x = (frame_shape[1] * astrometry_result['scale']) / 3600  # degrees
    fov_y = (frame_shape[0] * astrometry_result['scale']) / 3600  # degrees

    # Generate star positions JavaScript array
    star_data = []
    for i, center in enumerate(centers[1:], 1):
        star_data.append(f"{{id: {i}, x: {center[0]:.2f}, y: {center[1]:.2f}}}")
    stars_js = ",\n            ".join(star_data)

    # Replace placeholders in template
    replacements = {
        '{{RA_HMS}}': f'{ra_hours:02d}h {ra_minutes:02d}m {ra_seconds:04.1f}s',
        '{{RA_DEG}}': f'{ra_deg:.3f}°',
        '{{DEC_DMS}}': f'{dec_sign}{dec_degrees:02d}° {dec_arcmin:02d}\' {dec_arcsec:04.1f}"',
        '{{DEC_DEG}}': f'{dec_deg:.3f}°',
        '{{SCALE}}': f'{astrometry_result["scale"]:.2f} "/pixel',
        '{{ORIENTATION}}': f'{astrometry_result["orientation"]:.1f}°',
        '{{FOV}}': f'{fov_x:.2f}° × {fov_y:.2f}°',
        '{{NUM_STARS}}': str(num_stars),
        '{{ROLL}}': f'{astrometry_result["orientation"]:.1f}°',
        '{{PITCH}}': f'{dec_deg:.1f}°',
        '{{YAW}}': f'{ra_deg:.1f}°',
        '{{STARS_DATA}}': stars_js,
        '{{ORIENTATION_VALUE}}': f'{astrometry_result["orientation"]:.1f}',
        '{{FRAME_WIDTH}}': str(frame_shape[1]),
        '{{FRAME_HEIGHT}}': str(frame_shape[0]),
        '{{DEC_VALUE}}': f'{dec_deg:.1f}'
    }

    html_content = html_template
    for placeholder, value in replacements.items():
        html_content = html_content.replace(placeholder, value)

    output_file = os.path.join(dataset_dir, 'star_tracker_visualization.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML visualization saved to: {output_file}")
    print(f"Open in browser to view interactive 3D visualization!")
    return output_file


def main():
    """Main execution function."""
    # Configuration: Adjust this threshold to control star detection sensitivity
    # Higher = fewer stars (only brightest), Lower = more stars (includes dimmer ones)
    # 99.5 = more stars, 99.8 = balanced, 99.9 = only brightest stars
    STAR_DETECTION_THRESHOLD = 99.90

    # Step 1: Download dataset
    stars_path, dataset_dir = download_dataset()

    # Step 2: Load event stream
    events, width, height = load_event_stream(stars_path)

    # Step 2.5: Save basic star map before corrections
    save_basic_star_map(events, width, height, dataset_dir)

    # Step 3: Filter noise from events
    filtered_events = filter_noise(events, width, height, time_window=10000)

    # Step 4: Warp events to compensate for sidereal motion
    warped_events, warped_width, warped_height = warp_events(
        filtered_events, width, height, vx=-1.861, vy=0.549
    )

    # Step 5: Accumulate warped events and apply median filter
    filtered_frame = accumulate_and_filter_frame(warped_events, warped_width, warped_height)

    # Step 6: Detect and label individual stars
    labels_array, num_stars, binary_mask = detect_and_label_stars(
        filtered_frame, percentile_threshold=STAR_DETECTION_THRESHOLD
    )

    # Step 7: Calculate star centers
    centers, labelled_events = calculate_star_centers(
        warped_events, labels_array, num_stars, warped_height
    )

    # Step 8: Save binary image for API submission
    api_image_path = save_binary_image_for_api(filtered_frame, dataset_dir)

    # Step 9: Visualize detected stars
    output_path = visualize_detected_stars(filtered_frame, centers, num_stars, dataset_dir)

    # Step 10: Save detected stars data
    data_file = save_detected_stars_data(centers, num_stars, dataset_dir)

    # Step 11: Perform astrometry via API
    astrometry_result = perform_astrometry_via_api(dataset_dir)

    # Step 12: Save astrometry results if found
    if astrometry_result is not None:
        save_astrometry_results(astrometry_result, centers, num_stars, dataset_dir)
        # Step 13: Generate HTML visualization
        generate_html_visualization(astrometry_result, centers, num_stars, filtered_frame.shape, dataset_dir)

    print("\n" + "="*60)
    print("STAR MAPPING COMPLETE!")
    print("="*60)
    print(f"Detection threshold: {STAR_DETECTION_THRESHOLD}%")
    print(f"Total stars detected: {num_stars}")
    if astrometry_result is not None:
        print(f"\nSky Position:")
        print(f"  - RA:  {astrometry_result['ra']:.4f}°")
        print(f"  - DEC: {astrometry_result['dec']:.4f}°")
        print(f"  - Scale: {astrometry_result['scale']:.4f} arcsec/pixel")
    print(f"\nOutputs:")
    print(f"  - Star Map:  {output_path}")
    print(f"  - Star Data: {data_file}")
    print(f"\nTip: To adjust detection, change STAR_DETECTION_THRESHOLD")
    print(f"     Higher (99.9) = fewer/brighter stars only")
    print(f"     Lower (99.5) = more/dimmer stars included")
    print("="*60)


if __name__ == "__main__":
    main()
