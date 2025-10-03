from flask import Flask, request, jsonify
import numpy as np
import cv2
import astrometry
import json
import time
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Initialize solver globally to reuse cache
SOLVER = None


def get_solver():
    """Initialize and cache the astrometry solver."""
    global SOLVER
    if SOLVER is None:
        SOLVER = astrometry.Solver(
            astrometry.series_5200.index_files(
                cache_directory="astrometry_cache",
                scales={6},
            )
        )
    return SOLVER


def convert_frame_to_binary(frame, min_area=1, max_area=300):
    """
    Convert frame to pure binary image, keeping only star clusters and removing noise.
    Also detects star positions.

    Args:
        frame: Input frame (numpy array)
        min_area: Minimum pixel area for a cluster to be kept
        max_area: Maximum pixel area for a cluster to be kept

    Returns:
        Binary frame with only star clusters (numpy array)
        List of star positions [(star_id, x, y), ...]
    """
    if frame.max() == 0:
        return frame, []

    _, binary_frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frame, connectivity=8)

    filtered_frame = np.zeros_like(binary_frame)
    star_positions = []

    star_id = 1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered_frame[labels == i] = 255
            x, y = centroids[i]
            star_positions.append((star_id, x, y))
            star_id += 1

    return filtered_frame, star_positions


def solve_astrometry(star_positions, timeout_seconds=20):
    """
    Solve astrometry for star positions.

    Args:
        star_positions: List of (star_id, x, y) tuples
        timeout_seconds: Maximum time to wait for solution

    Returns:
        Astrometry result dict or None
    """
    try:
        solver = get_solver()
        stars = [[float(x), float(y)] for _, x, y in star_positions]

        solution = solver.solve(
            stars=stars,
            size_hint=None,
            position_hint=None,
            solution_parameters=astrometry.SolutionParameters(),
        )

        if solution.has_match():
            match = solution.best_match()
            return {
                'ra': match.center_ra_deg,
                'dec': match.center_dec_deg,
                'scale': match.scale_arcsec_per_pixel,
                'orientation': 0
            }
        else:
            return None

    except Exception as e:
        print(f"Solver error: {e}")
        return None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'astrometry-solver'}), 200


@app.route('/solve', methods=['POST'])
def solve():
    """
    Solve astrometry for an uploaded image.

    Request JSON format:
    {
        "image": "base64_encoded_image_string",
        "min_area": 1,  # optional
        "max_area": 300,  # optional
        "timeout": 20  # optional
    }

    Response JSON format:
    {
        "status": "success" | "failed" | "error",
        "num_stars": int,
        "stars": [[star_id, x, y], ...],
        "ra_deg": float,
        "dec_deg": float,
        "scale_arcsec_per_pixel": float,
        "orientation_deg": float,
        "solve_time_seconds": float,
        "message": str  # optional error message
    }
    """
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400

        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(BytesIO(image_data))
            frame = np.array(image.convert('L'))  # Convert to grayscale
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid image data: {str(e)}'
            }), 400

        # Get optional parameters
        min_area = data.get('min_area', 1)
        max_area = data.get('max_area', 300)
        timeout = data.get('timeout', 20)

        # Convert to binary and detect stars
        binary_frame, star_positions = convert_frame_to_binary(frame, min_area, max_area)

        if len(star_positions) == 0:
            solve_time = time.time() - start_time
            return jsonify({
                'status': 'failed',
                'num_stars': 0,
                'stars': [],
                'message': 'No stars detected in image',
                'solve_time_seconds': round(solve_time, 2)
            }), 200

        # Format star positions
        stars_array = [[star_id, round(x, 2), round(y, 2)] for star_id, x, y in star_positions]

        # Solve astrometry
        result = solve_astrometry(star_positions, timeout)
        solve_time = time.time() - start_time

        if result is not None:
            return jsonify({
                'status': 'success',
                'num_stars': len(star_positions),
                'stars': stars_array,
                'ra_deg': round(result['ra'], 6),
                'dec_deg': round(result['dec'], 6),
                'scale_arcsec_per_pixel': round(result['scale'], 6),
                'orientation_deg': round(result['orientation'], 6),
                'solve_time_seconds': round(solve_time, 2)
            }), 200
        else:
            return jsonify({
                'status': 'failed',
                'num_stars': len(star_positions),
                'stars': stars_array,
                'message': 'Could not solve astrometry',
                'solve_time_seconds': round(solve_time, 2)
            }), 200

    except Exception as e:
        solve_time = time.time() - start_time
        return jsonify({
            'status': 'error',
            'message': str(e),
            'solve_time_seconds': round(solve_time, 2)
        }), 500


@app.route('/solve/stars', methods=['POST'])
def solve_from_stars():
    """
    Solve astrometry from pre-detected star positions.

    Request JSON format:
    {
        "stars": [[star_id, x, y], ...],
        "timeout": 20  # optional
    }

    Response JSON format:
    {
        "status": "success" | "failed" | "error",
        "num_stars": int,
        "ra_deg": float,
        "dec_deg": float,
        "scale_arcsec_per_pixel": float,
        "orientation_deg": float,
        "solve_time_seconds": float,
        "message": str  # optional error message
    }
    """
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or 'stars' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No star data provided'
            }), 400

        stars = data['stars']
        timeout = data.get('timeout', 20)

        if len(stars) == 0:
            return jsonify({
                'status': 'failed',
                'num_stars': 0,
                'message': 'No stars provided',
                'solve_time_seconds': 0
            }), 200

        # Convert to star positions format
        star_positions = [(star[0], star[1], star[2]) for star in stars]

        # Solve astrometry
        result = solve_astrometry(star_positions, timeout)
        solve_time = time.time() - start_time

        if result is not None:
            return jsonify({
                'status': 'success',
                'num_stars': len(star_positions),
                'ra_deg': round(result['ra'], 6),
                'dec_deg': round(result['dec'], 6),
                'scale_arcsec_per_pixel': round(result['scale'], 6),
                'orientation_deg': round(result['orientation'], 6),
                'solve_time_seconds': round(solve_time, 2)
            }), 200
        else:
            return jsonify({
                'status': 'failed',
                'num_stars': len(star_positions),
                'message': 'Could not solve astrometry',
                'solve_time_seconds': round(solve_time, 2)
            }), 200

    except Exception as e:
        solve_time = time.time() - start_time
        return jsonify({
            'status': 'error',
            'message': str(e),
            'solve_time_seconds': round(solve_time, 2)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
