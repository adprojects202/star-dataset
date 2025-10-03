import urllib.request
import os
import numpy
import event_stream
import cv2
import requests
import json
import time
import csv
from dotenv import load_dotenv
from scipy import ndimage
import matplotlib.pyplot as plt


def create_output_folders(dataset_dir):
    """Create organized folder structure for outputs."""
    folders = {
        'raw_frames': os.path.join(dataset_dir, 'output', '1_raw_frames'),
        'warped_frames': os.path.join(dataset_dir, 'output', '2_warped_frames'),
        'enhanced_frames': os.path.join(dataset_dir, 'output', '3_enhanced_frames'),
        'debug': os.path.join(dataset_dir, 'output', 'debug'),
        'results': os.path.join(dataset_dir, 'output', 'results')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    print(f"\nCreated output folder structure in: {os.path.join(dataset_dir, 'output')}")
    return folders


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

    events = events.view(dtype=events.dtype.descr[:3] + [(('on', 'p'), '?')])
    return events, width, height


def split_events_into_time_windows(events, window_duration_ms=1000, overlap_ms=500):
    """Split events into overlapping time windows."""
    print(f"\nSplitting events into time windows:")
    print(f"  Window duration: {window_duration_ms}ms")
    print(f"  Overlap: {overlap_ms}ms")
    
    t_start = events["t"][0]
    t_end = events["t"][-1]
    total_duration_ms = (t_end - t_start) / 1000
    
    window_duration_us = window_duration_ms * 1000
    overlap_us = overlap_ms * 1000
    step_us = window_duration_us - overlap_us
    
    num_windows = int((t_end - t_start - window_duration_us) / step_us) + 1
    
    print(f"  Total duration: {total_duration_ms:.2f}ms")
    print(f"  Number of windows: {num_windows}")
    
    windows = []
    
    for i in range(num_windows):
        window_start = t_start + i * step_us
        window_end = window_start + window_duration_us
        window_center = (window_start + window_end) / 2
        
        mask = (events["t"] >= window_start) & (events["t"] < window_end)
        window_events = events[mask]
        
        if len(window_events) > 0:
            windows.append((window_events, window_start, window_center))
    
    print(f"  Generated {len(windows)} windows")
    return windows


def estimate_local_motion(window_events, num_blocks=10, max_velocity=50.0):
    """
    Estimate motion within a single time window with safety limits.
    
    Args:
        window_events: Events in the window
        num_blocks: Number of blocks to divide time into
        max_velocity: Maximum allowed velocity in px/s (safety limit)
    """
    if len(window_events) < 100:
        return 0.0, 0.0
    
    t_start = window_events["t"][0]
    t_end = window_events["t"][-1]
    duration_us = t_end - t_start
    
    if duration_us < 10000:  # Less than 10ms - too short
        return 0.0, 0.0
    
    block_duration = duration_us // num_blocks
    
    centroids_x = []
    centroids_y = []
    times = []
    
    for i in range(num_blocks):
        block_start = t_start + i * block_duration
        block_end = block_start + block_duration
        
        mask = (window_events["t"] >= block_start) & (window_events["t"] < block_end)
        block_events = window_events[mask]
        
        if len(block_events) > 5:
            cx = numpy.mean(block_events["x"])
            cy = numpy.mean(block_events["y"])
            centroids_x.append(cx)
            centroids_y.append(cy)
            times.append((block_start + block_end) / 2)
    
    if len(centroids_x) < 5:
        return 0.0, 0.0
    
    times_array = numpy.array(times)
    times_normalized = (times_array - times_array[0]) / 1e6
    
    # Fit linear motion
    vx = numpy.polyfit(times_normalized, centroids_x, 1)[0]
    vy = numpy.polyfit(times_normalized, centroids_y, 1)[0]
    
    # Apply safety limits
    vx = numpy.clip(vx, -max_velocity, max_velocity)
    vy = numpy.clip(vy, -max_velocity, max_velocity)
    
    return vx, vy


def warp_window_events(window_events, width, height, vx, vy, max_dimension=2048):
    """
    Warp events within a window with strict size limits.
    
    Args:
        window_events: Events to warp
        width, height: Original dimensions
        vx, vy: Velocity in px/s
        max_dimension: Maximum allowed dimension (safety limit)
    """
    if len(window_events) == 0:
        return window_events, width, height
    
    t_center = (window_events["t"][0] + window_events["t"][-1]) / 2
    
    warped_events = numpy.zeros(len(window_events), dtype=window_events.dtype)
    warped_events["t"] = window_events["t"]
    warped_events["on"] = window_events["on"]
    
    # Calculate displacement relative to center time
    time_offset = (window_events["t"] - t_center) / 1e6  # seconds
    
    # Apply warping
    warped_x = window_events["x"] - (vx * time_offset)
    warped_y = window_events["y"] - (vy * time_offset)
    
    # Find bounds
    x_min, x_max = warped_x.min(), warped_x.max()
    y_min, y_max = warped_y.min(), warped_y.max()
    
    # Shift to positive coordinates
    warped_x = warped_x - x_min
    warped_y = warped_y - y_min
    
    # Calculate new dimensions
    new_width = int(numpy.ceil(x_max - x_min)) + 1
    new_height = int(numpy.ceil(y_max - y_min)) + 1
    
    # Safety check - limit maximum size
    if new_width > max_dimension or new_height > max_dimension:
        print(f"  WARNING: Warped size too large ({new_width}x{new_height}), skipping warp")
        # Return original events without warping
        warped_events["x"] = window_events["x"]
        warped_events["y"] = window_events["y"]
        return warped_events, width, height
    
    # Assign warped coordinates
    warped_events["x"] = warped_x.astype(numpy.int16)
    warped_events["y"] = warped_y.astype(numpy.int16)
    
    # Filter valid events
    valid_mask = (
        (warped_events["x"] >= 0) & 
        (warped_events["x"] < new_width) &
        (warped_events["y"] >= 0) & 
        (warped_events["y"] < new_height)
    )
    
    return warped_events[valid_mask], new_width, new_height


def accumulate_window_to_frame(window_events, width, height):
    """Accumulate events from a single window into a frame."""
    frame = numpy.zeros((height, width), dtype=numpy.float32)
    
    for event in window_events:
        x, y = event["x"], event["y"]
        if 0 <= x < width and 0 <= y < height:
            frame[height - 1 - y, x] += 1
    
    return frame


def save_debug_frame(frame, folder, filename):
    """Save debug frame with normalization."""
    if frame.max() > 0:
        normalized = (frame / frame.max() * 255).astype(numpy.uint8)
    else:
        normalized = numpy.zeros_like(frame, dtype=numpy.uint8)
    
    output_path = os.path.join(folder, filename)
    cv2.imwrite(output_path, normalized)
    return output_path


def enhance_frame(frame, gaussian_sigma=1.5, gamma=0.7):
    """Enhance frame for star detection."""
    if frame.max() == 0:
        return numpy.zeros_like(frame, dtype=numpy.uint8)
    
    # Gaussian smoothing
    smoothed = ndimage.gaussian_filter(frame, sigma=gaussian_sigma)
    
    # Normalize
    normalized = smoothed / smoothed.max() if smoothed.max() > 0 else smoothed
    
    # Apply gamma correction
    enhanced = numpy.power(normalized, gamma)
    
    # Convert to uint8
    final = (enhanced * 255).astype(numpy.uint8)
    
    return final


def count_bright_spots(frame, threshold_percentile=98.0):
    """Count bright spots above threshold."""
    if frame.max() == 0:
        return 0
    
    non_zero = frame[frame > 0]
    if len(non_zero) == 0:
        return 0
    
    threshold = numpy.percentile(non_zero, threshold_percentile)
    binary = (frame > threshold).astype(numpy.uint8)
    
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 2 <= area <= 500:
            count += 1
    
    return count


def submit_to_astrometry_api(image_path, timeout_seconds=90):
    """Submit image to Astrometry.net API."""
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

    print("  API key found. Submitting to Astrometry.net...")
    try:
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
            print(f"Response: {response.text}")
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


def process_windows_for_trajectory(windows, width, height, folders):
    """Process each time window to build trajectory with incremental CSV saving."""
    print("\n" + "="*60)
    print("PROCESSING TIME WINDOWS FOR TRAJECTORY")
    print("="*60)
    
    # Open CSV file at start
    output_csv = os.path.join(folders['results'], 'star_tracker_trajectory.csv')
    csvfile = open(output_csv, 'w', newline='')
    fieldnames = ['frame_number', 'num_stars', 'stars_data', 'ra_deg', 'dec_deg', 
                  'scale_arcsec_per_pixel', 'orientation_deg', 'status', 'solve_time_seconds']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()  # Ensure header is written immediately
    
    print(f"Saving results incrementally to: {output_csv}\n")
    
    trajectory = []
    
    try:
        for window_idx, (window_events, window_start, window_center) in enumerate(windows):
            print(f"\nWindow {window_idx + 1}/{len(windows)}")
            print(f"  Time: {window_center/1e6:.3f}s")
            print(f"  Events: {len(window_events)}")
            
            # Stage 1: Raw accumulation (no motion compensation)
            raw_frame = accumulate_window_to_frame(window_events, width, height)
            raw_path = save_debug_frame(raw_frame, folders['raw_frames'], 
                                        f'window_{window_idx:03d}_raw.png')
            print(f"  Saved raw frame: {os.path.basename(raw_path)}")
            
            # Stage 2: Estimate and apply motion compensation
            vx, vy = estimate_local_motion(window_events, num_blocks=10, max_velocity=20.0)
            print(f"  Motion: vx={vx:.3f}, vy={vy:.3f} px/s")
            
            warped_events, new_width, new_height = warp_window_events(
                window_events, width, height, vx, vy, max_dimension=2048
            )
            print(f"  Warped: {len(warped_events)} events, size {new_width}x{new_height}")
            
            warped_frame = accumulate_window_to_frame(warped_events, new_width, new_height)
            warped_path = save_debug_frame(warped_frame, folders['warped_frames'],
                                          f'window_{window_idx:03d}_warped.png')
            print(f"  Saved warped frame: {os.path.basename(warped_path)}")
            
            # Stage 3: Enhancement
            enhanced = enhance_frame(warped_frame, gaussian_sigma=1.5, gamma=0.7)
            enhanced_path = os.path.join(folders['enhanced_frames'], 
                                        f'window_{window_idx:03d}_enhanced.png')
            cv2.imwrite(enhanced_path, enhanced)
            print(f"  Saved enhanced frame: {os.path.basename(enhanced_path)}")
            
            # Count stars
            num_stars = count_bright_spots(enhanced, threshold_percentile=98.0)
            print(f"  Stars detected: {num_stars}")
            
            # Collect star data
            stars_data = []
            if num_stars > 0:
                non_zero_coords = numpy.argwhere(enhanced > numpy.percentile(enhanced, 98))
                for i, (y, x) in enumerate(non_zero_coords[:num_stars]):
                    stars_data.append([i + 1, float(x), float(y)])
            
            # Submit to API if enough stars
            result_dict = {
                'frame_number': window_idx,
                'num_stars': num_stars,
                'stars_data': json.dumps(stars_data),
                'ra_deg': None,
                'dec_deg': None,
                'scale_arcsec_per_pixel': None,
                'orientation_deg': None,
                'status': 'insufficient_stars',
                'solve_time_seconds': None
            }
            
            if num_stars >= 5:
                print(f"  Submitting to API...")
                start_time = time.time()
                result = submit_to_astrometry_api(enhanced_path, timeout_seconds=90)
                solve_time = time.time() - start_time
                
                if result:
                    result_dict.update({
                        'ra_deg': result['ra'],
                        'dec_deg': result['dec'],
                        'scale_arcsec_per_pixel': result['scale'],
                        'orientation_deg': result['orientation'],
                        'status': 'success',
                        'solve_time_seconds': round(solve_time, 2)
                    })
                    print(f"  ✓ SUCCESS: RA={result['ra']:.4f}°, DEC={result['dec']:.4f}°")
                else:
                    result_dict['status'] = 'failed'
                    result_dict['solve_time_seconds'] = round(solve_time, 2)
                    print(f"  ✗ FAILED: Could not solve")
            else:
                print(f"  ⊘ SKIPPED: Insufficient stars ({num_stars} < 5)")
            
            # Add to trajectory list
            trajectory.append(result_dict)
            
            # Write this result to CSV immediately
            writer.writerow(result_dict)
            csvfile.flush()  # Force write to disk
            
    finally:
        # Close CSV file
        csvfile.close()
        print(f"\nCSV file closed: {output_csv}")
    
    return trajectory
def save_trajectory_to_csv(trajectory, folders):
    """Save trajectory data to CSV."""
    output_csv = os.path.join(folders['results'], 'star_tracker_trajectory.csv')
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['window', 'time_us', 'time_s', 'num_stars', 'ra_deg', 'dec_deg', 
                     'scale', 'orientation_deg', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for point in trajectory:
            writer.writerow(point)
    
    print(f"\nTrajectory saved to: {output_csv}")
    return output_csv


def plot_trajectory(trajectory, folders):
    """Plot the trajectory over time."""
    print("\nGenerating trajectory plots...")
    
    success_points = [p for p in trajectory if p['status'] == 'success']
    
    if len(success_points) == 0:
        print("  No successful solutions to plot")
        return
    
    times = [p['time_s'] for p in success_points]
    ras = [p['ra_deg'] for p in success_points]
    decs = [p['dec_deg'] for p in success_points]
    orientations = [p['orientation_deg'] for p in success_points]
    
    # Time series plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(times, ras, 'o-', linewidth=2, markersize=8)
    axes[0].set_ylabel('Right Ascension (deg)', fontsize=12)
    axes[0].set_title('Star Tracker Trajectory', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times, decs, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_ylabel('Declination (deg)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times, orientations, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Orientation (deg)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(folders['results'], 'trajectory_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Trajectory plot saved to: {output_path}")
    plt.close()
    
    # Sky path plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(ras, decs, c=times, cmap='viridis', s=100, edgecolors='black')
    ax.plot(ras, decs, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Right Ascension (deg)', fontsize=12)
    ax.set_ylabel('Declination (deg)', fontsize=12)
    ax.set_title('Sky Path (RA vs DEC)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)', fontsize=11)
    
    output_path = os.path.join(folders['results'], 'sky_path.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Sky path plot saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*60)
    print("STAR TRACKER - REAL-TIME TRAJECTORY WITH DEBUG")
    print("="*60)

    # Configuration
    WINDOW_DURATION_MS = 1000
    OVERLAP_MS = 500

    # Step 1: Download dataset
    stars_path, dataset_dir = download_dataset()

    # Step 2: Create output folders
    folders = create_output_folders(dataset_dir)

    # Step 3: Load event stream
    events, width, height = load_event_stream(stars_path)

    # Step 4: Split into time windows
    windows = split_events_into_time_windows(
        events, 
        window_duration_ms=WINDOW_DURATION_MS,
        overlap_ms=OVERLAP_MS
    )

    # Step 5: Process each window (CSV saved incrementally inside)
    trajectory = process_windows_for_trajectory(windows, width, height, folders)

    # Step 6: Plot trajectory (if any successful results)
    plot_trajectory(trajectory, folders)

    # Print summary
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS COMPLETE!")
    print("="*60)
    
    success_count = sum(1 for p in trajectory if p['status'] == 'success')
    print(f"Total windows: {len(trajectory)}")
    print(f"Successful solutions: {success_count}")
    print(f"Success rate: {100*success_count/len(trajectory) if len(trajectory) > 0 else 0:.1f}%")
    
    print(f"\nOutput structure:")
    print(f"  Raw frames: {folders['raw_frames']}")
    print(f"  Warped frames: {folders['warped_frames']}")
    print(f"  Enhanced frames: {folders['enhanced_frames']}")
    print(f"  Results: {folders['results']}")
    print("="*60)

if __name__ == "__main__":
    main()