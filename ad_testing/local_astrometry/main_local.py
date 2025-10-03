import urllib.request
import os
import event_stream
import numpy
import cv2
import csv
import json
import time
import base64
import requests
from io import BytesIO
from PIL import Image


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


def convert_events_to_frames(events, width, height, accumulation_time_ms=1000, overlap_ms=0):
    """
    Convert events into frames by accumulating events over time windows.

    Args:
        events: Event array
        width: Frame width
        height: Frame height
        accumulation_time_ms: Time window to accumulate events (larger = more events per frame)
        overlap_ms: Overlap between consecutive frames (0 = no overlap)

    Returns:
        List of frames (numpy arrays)
    """
    print(f"\nConverting events to frames (accumulation: {accumulation_time_ms}ms, overlap: {overlap_ms}ms)...")

    t_start = events["t"][0]
    t_end = events["t"][-1]
    total_duration_ms = (t_end - t_start) / 1000
    accumulation_time_us = accumulation_time_ms * 1000
    overlap_us = overlap_ms * 1000
    step_us = accumulation_time_us - overlap_us

    num_frames = int((t_end - t_start - accumulation_time_us) / step_us) + 1
    if num_frames < 1:
        num_frames = 1

    print(f"Total duration: {total_duration_ms:.2f}ms")
    print(f"Number of frames: {num_frames}")

    frames = []
    for frame_idx in range(num_frames):
        frame_start = t_start + frame_idx * step_us
        frame_end = frame_start + accumulation_time_us

        mask = (events["t"] >= frame_start) & (events["t"] < frame_end)
        frame_events = events[mask]

        frame = numpy.zeros((height, width), dtype=numpy.float32)
        if len(frame_events) > 0:
            for event in frame_events:
                x, y = event["x"], event["y"]
                if 0 <= x < width and 0 <= y < height:
                    frame[height - 1 - y, x] += 1

        if frame.max() > 0:
            frame = (frame / frame.max() * 255).astype(numpy.uint8)
        else:
            frame = frame.astype(numpy.uint8)

        frames.append(frame)
        print(f"Frame {frame_idx}: {len(frame_events)} events, max intensity: {frame.max()}")

    print(f"Generated {len(frames)} frames")
    return frames


def frame_to_base64(frame):
    """Convert numpy frame to base64 encoded string."""
    img = Image.fromarray(frame)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def solve_frame_via_api(frame, api_url="http://localhost:5000/solve", min_area=1, max_area=300, timeout=20):
    """
    Solve astrometry for a frame using the Flask API.

    Args:
        frame: Numpy array of the frame
        api_url: URL of the astrometry API endpoint
        min_area: Minimum pixel area for star detection
        max_area: Maximum pixel area for star detection
        timeout: Timeout in seconds

    Returns:
        API response dict or None
    """
    try:
        image_b64 = frame_to_base64(frame)

        payload = {
            'image': image_b64,
            'min_area': min_area,
            'max_area': max_area,
            'timeout': timeout
        }

        response = requests.post(api_url, json=payload, timeout=timeout + 10)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"API request timeout after {timeout + 10} seconds")
        return None
    except Exception as e:
        print(f"API request error: {e}")
        return None


def process_frames_and_save_results(frames, dataset_dir, api_url="http://localhost:5000/solve", output_csv=None):
    """
    Process all frames using the API and save results to CSV.

    Args:
        frames: List of frames
        dataset_dir: Directory for outputs
        api_url: URL of the astrometry API
        output_csv: Path to output CSV file

    Returns:
        Path to CSV file
    """
    if output_csv is None:
        output_csv = os.path.join(dataset_dir, 'astrometry_results_api.csv')

    print(f"\nProcessing {len(frames)} frames with API at {api_url}...")
    print(f"Output CSV: {output_csv}")

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['frame_number', 'num_stars', 'stars_data', 'ra_deg', 'dec_deg', 'scale_arcsec_per_pixel', 'orientation_deg', 'status', 'solve_time_seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for frame_idx, frame in enumerate(frames):
            print(f"\n[{frame_idx+1}/{len(frames)}] Processing frame {frame_idx}")
            start_time = time.time()

            try:
                result = solve_frame_via_api(frame, api_url)
                solve_time = time.time() - start_time

                if result is None:
                    writer.writerow({
                        'frame_number': frame_idx,
                        'num_stars': 0,
                        'stars_data': '[]',
                        'ra_deg': '',
                        'dec_deg': '',
                        'scale_arcsec_per_pixel': '',
                        'orientation_deg': '',
                        'status': 'api_error',
                        'solve_time_seconds': f'{solve_time:.2f}'
                    })
                    print(f"API ERROR - Time: {solve_time:.2f}s")
                    csvfile.flush()
                    continue

                status = result.get('status', 'unknown')

                if status == 'success':
                    writer.writerow({
                        'frame_number': frame_idx,
                        'num_stars': result['num_stars'],
                        'stars_data': json.dumps(result['stars']),
                        'ra_deg': f'{result["ra_deg"]:.6f}',
                        'dec_deg': f'{result["dec_deg"]:.6f}',
                        'scale_arcsec_per_pixel': f'{result["scale_arcsec_per_pixel"]:.6f}',
                        'orientation_deg': f'{result["orientation_deg"]:.6f}',
                        'status': 'success',
                        'solve_time_seconds': f'{result["solve_time_seconds"]:.2f}'
                    })
                    print(f"SUCCESS - RA: {result['ra_deg']:.4f}, DEC: {result['dec_deg']:.4f}, Time: {result['solve_time_seconds']:.2f}s")
                else:
                    writer.writerow({
                        'frame_number': frame_idx,
                        'num_stars': result.get('num_stars', 0),
                        'stars_data': json.dumps(result.get('stars', [])),
                        'ra_deg': '',
                        'dec_deg': '',
                        'scale_arcsec_per_pixel': '',
                        'orientation_deg': '',
                        'status': status,
                        'solve_time_seconds': f'{result.get("solve_time_seconds", 0):.2f}'
                    })
                    print(f"{status.upper()} - {result.get('message', 'No message')}, Time: {result.get('solve_time_seconds', 0):.2f}s")

                csvfile.flush()

            except Exception as e:
                solve_time = time.time() - start_time
                print(f"ERROR - {str(e)}")
                writer.writerow({
                    'frame_number': frame_idx,
                    'num_stars': 0,
                    'stars_data': '[]',
                    'ra_deg': '',
                    'dec_deg': '',
                    'scale_arcsec_per_pixel': '',
                    'orientation_deg': '',
                    'status': f'error: {str(e)}',
                    'solve_time_seconds': f'{solve_time:.2f}'
                })
                csvfile.flush()

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_csv}")
    print("="*60)

    return output_csv


def main():
    """Main execution function."""
    print("="*60)
    print("STAR ASTROMETRY PROCESSING - API CLIENT")
    print("="*60)

    api_url = os.getenv('ASTROMETRY_API_URL', 'http://localhost:5000/solve')
    print(f"Using API at: {api_url}")

    stars_path, dataset_dir = download_dataset()
    events, width, height = load_event_stream(stars_path)
    frames = convert_events_to_frames(events, width, height, accumulation_time_ms=200, overlap_ms=100)
    output_csv = process_frames_and_save_results(frames, dataset_dir, api_url)

    print(f"\nAll results saved to: {output_csv}")


if __name__ == "__main__":
    main()
