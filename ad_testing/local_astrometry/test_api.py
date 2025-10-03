import requests
import base64
import json
from PIL import Image
from io import BytesIO
import numpy as np


def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get('http://localhost:5000/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_solve_with_image():
    """Test the /solve endpoint with a sample image."""
    print("Testing /solve endpoint with image...")

    # Create a simple test image with some bright spots (simulated stars)
    img_array = np.zeros((640, 480), dtype=np.uint8)

    # Add some "stars" (bright spots)
    stars = [
        (100, 100), (200, 150), (300, 200), (400, 250),
        (150, 300), (250, 350), (350, 400), (100, 450)
    ]

    for x, y in stars:
        # Create a small bright region for each star
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= x+dx < 480 and 0 <= y+dy < 640:
                    img_array[y+dy, x+dx] = 255

    # Convert to base64
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    # Make request
    payload = {
        'image': img_b64,
        'min_area': 1,
        'max_area': 300,
        'timeout': 20
    }

    response = requests.post('http://localhost:5000/solve', json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_solve_with_stars():
    """Test the /solve/stars endpoint with pre-detected stars."""
    print("Testing /solve/stars endpoint...")

    # Sample star positions
    stars = [
        [1, 100.5, 100.2],
        [2, 200.3, 150.8],
        [3, 300.1, 200.5],
        [4, 400.9, 250.3],
        [5, 150.2, 300.7],
        [6, 250.8, 350.1],
        [7, 350.4, 400.9],
        [8, 100.6, 450.2]
    ]

    payload = {
        'stars': stars,
        'timeout': 20
    }

    response = requests.post('http://localhost:5000/solve/stars', json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


if __name__ == '__main__':
    print("="*60)
    print("ASTROMETRY API TEST SUITE")
    print("="*60)
    print()

    try:
        test_health()
        test_solve_with_image()
        test_solve_with_stars()

        print("="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API at http://localhost:5000")
        print("Make sure the Docker container is running:")
        print("  cd ad_testing/local_astrometry")
        print("  docker-compose up")
    except Exception as e:
        print(f"ERROR: {e}")
