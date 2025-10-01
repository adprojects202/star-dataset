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


def download_dataset():
    """Download the star dataset from GitHub to the dataset folder."""
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    stars_path = os.path.join(dataset_dir, 'stars_cut.es')

    print("Checking for star dataset...")
    if not os.path.exists(stars_path):
        print("Downloading star dataset...")
        url = "https://github.com/neuromorphicsystems/tutorials/raw/main/data/stars_cut.es"
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


def detect_and_label_stars(filtered_frame):
    """Detect and label individual stars in the frame."""
    print("\nDetecting and labeling stars...")

    # Create binary mask using 99.5% percentile
    nonzero = filtered_frame[filtered_frame > 0.0]
    threshold = numpy.percentile(nonzero, 99.5)
    binary_mask = filtered_frame > threshold

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


def visualize_star_map(filtered_frame, centers, num_stars, dataset_dir):
    """Visualize the final star map with detected stars."""
    print("\nVisualizing star map...")

    plt.figure(figsize=(12, 8))
    plt.imshow(
        filtered_frame,
        norm=matplotlib.colors.LogNorm(),
        cmap="magma",
    )

    # Plot star centers
    plt.scatter(
        x=centers[1:, 0],
        y=filtered_frame.shape[0] - 1 - centers[1:, 1],
        marker="o",
        facecolor="none",
        edgecolors="#00ff00",
        linewidth=1.0,
        s=100,
    )

    plt.title(f"Star Map - {num_stars} Stars Detected")
    plt.colorbar(label="Event Count (log scale)")

    output_path = os.path.join(dataset_dir, 'star_map_final.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Star map saved to: {output_path}")
    plt.show()

    return output_path


def main():
    """Main execution function."""
    # Step 1: Download dataset
    stars_path, dataset_dir = download_dataset()

    # Step 2: Load event stream
    events, width, height = load_event_stream(stars_path)

    # Step 3: Filter noise from events
    filtered_events = filter_noise(events, width, height, time_window=10000)

    # Step 4: Warp events to compensate for sidereal motion
    warped_events, warped_width, warped_height = warp_events(
        filtered_events, width, height, vx=-1.861, vy=0.549
    )

    # Step 5: Accumulate warped events and apply median filter
    filtered_frame = accumulate_and_filter_frame(warped_events, warped_width, warped_height)

    # Step 6: Detect and label individual stars
    labels_array, num_stars, binary_mask = detect_and_label_stars(filtered_frame)

    # Step 7: Calculate star centers
    centers, labelled_events = calculate_star_centers(
        warped_events, labels_array, num_stars, warped_height
    )

    # Step 8: Visualize final star map
    output_path = visualize_star_map(filtered_frame, centers, num_stars, dataset_dir)

    print("\n" + "="*60)
    print("STAR MAPPING COMPLETE!")
    print("="*60)
    print(f"Total stars detected: {num_stars}")
    print(f"Output saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
