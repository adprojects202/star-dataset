import csv
import json
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

csv.field_size_limit(10000000)


def load_csv_data(csv_path):
    """
    Load astrometry results from CSV file.

    Returns:
        List of frame data dictionaries
    """
    print(f"Loading data from: {csv_path}")

    frames_data = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Only process rows where status is 'success'
            if row['status'].strip().lower() == 'success':
                try:
                    stars_data = json.loads(row['stars_data'])

                    frames_data.append({
                        'frame_number': int(row['frame_number']),
                        'num_stars': int(row['num_stars']),
                        'stars': stars_data,
                        'ra': float(row['ra_deg']),
                        'dec': float(row['dec_deg']),
                        'scale': float(row['scale_arcsec_per_pixel']),
                        'orientation': float(row['orientation_deg']) if row['orientation_deg'] else 0,
                        'solve_time': float(row['solve_time_seconds']) if row.get('solve_time_seconds') else 0
                    })
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Skipping frame {row['frame_number']}: {e}")
                    continue
            else:
                print(f"Skipping frame {row['frame_number']}: status is '{row['status']}', not 'success'")

    print(f"Loaded {len(frames_data)} successful frames")
    return frames_data


def create_html_animation(frames_data, output_path='animation.html'):
    """
    Create HTML animated visualization cycling through frames.

    Args:
        frames_data: List of frame data dictionaries
        output_path: Path to save HTML file
    """
    if not frames_data:
        print("No data to visualize")
        return

    print(f"Creating HTML animation with {len(frames_data)} frames...")

    frames_json = json.dumps(frames_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Star Tracker Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
            border: 2px solid #00ff00;
            padding: 15px;
            background: rgba(0, 255, 0, 0.05);
        }}
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
        }}
        .visualization {{
            border: 2px solid #00ff00;
            background: #000;
            position: relative;
        }}
        canvas {{
            display: block;
            background: #000;
        }}
        .info-panel {{
            border: 2px solid #00ff00;
            padding: 15px;
            background: rgba(0, 255, 0, 0.05);
        }}
        .info-section {{
            margin-bottom: 20px;
            padding: 10px;
            border-left: 3px solid #00ff00;
            background: rgba(0, 255, 0, 0.02);
        }}
        .info-section h3 {{
            margin-top: 0;
            color: #00ffff;
        }}
        .data-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(0, 255, 0, 0.2);
        }}
        .label {{
            color: #00ff00;
        }}
        .value {{
            color: #ffff00;
            font-weight: bold;
        }}
        .compass {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 80px;
            height: 80px;
            border: 2px solid #00ff00;
            border-radius: 50%;
            background: rgba(0, 0, 0, 0.8);
        }}
        .controls {{
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            border: 2px solid #00ff00;
            background: rgba(0, 255, 0, 0.05);
        }}
        button {{
            background: #00ff00;
            color: #000;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
        }}
        button:hover {{
            background: #00ffff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>== STAR TRACKER ASTROMETRY VISUALIZATION ==</h1>
            <p>Field of View Analysis & Attitude Determination</p>
        </div>

        <div class="main-grid">
            <div class="visualization">
                <canvas id="starField" width="900" height="720"></canvas>
                <div class="compass" id="compass"></div>
            </div>

            <div class="info-panel">
                <div class="info-section">
                    <h3>📍 POINTING COORDINATES</h3>
                    <div class="data-row">
                        <span class="label">RA (deg):</span>
                        <span class="value" id="ra-deg">0.000</span>
                    </div>
                    <div class="data-row">
                        <span class="label">DEC (deg):</span>
                        <span class="value" id="dec-deg">0.000</span>
                    </div>
                </div>

                <div class="info-section">
                    <h3>🎯 IMAGE PROPERTIES</h3>
                    <div class="data-row">
                        <span class="label">Scale:</span>
                        <span class="value" id="scale">0.00</span>
                    </div>
                    <div class="data-row">
                        <span class="label">Orientation:</span>
                        <span class="value" id="orientation">0.0°</span>
                    </div>
                    <div class="data-row">
                        <span class="label">Stars Detected:</span>
                        <span class="value" id="num-stars">0</span>
                    </div>
                </div>

                <div class="info-section">
                    <h3>🛰️ SPACECRAFT ATTITUDE</h3>
                    <div class="data-row">
                        <span class="label">Roll:</span>
                        <span class="value" id="roll">0.0°</span>
                    </div>
                    <div class="data-row">
                        <span class="label">Pitch:</span>
                        <span class="value" id="pitch">0.0°</span>
                    </div>
                    <div class="data-row">
                        <span class="label">Yaw:</span>
                        <span class="value" id="yaw">0.0°</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="controls">
            <button id="prevBtn">Previous</button>
            <button id="playBtn">Play</button>
            <button id="pauseBtn">Pause</button>
            <button id="nextBtn">Next</button>
            <br><br>
            <span>Frame: <span id="frameCounter">0</span> / <span id="totalFrames">0</span></span>
        </div>
    </div>

    <script>
        const framesData = {frames_json};
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;

        const canvas = document.getElementById('starField');
        const ctx = canvas.getContext('2d');
        
        // Actual sensor resolution
        const SENSOR_WIDTH = 1280;
        const SENSOR_HEIGHT = 720;

        document.getElementById('totalFrames').textContent = framesData.length;

        function drawStarField(frameData) {{
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw grid
            ctx.strokeStyle = 'rgba(0, 255, 0, 0.2)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 10; i++) {{
                const x = (canvas.width / 10) * i;
                const y = (canvas.height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }}

            // Draw center crosshairs
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            ctx.strokeStyle = '#00ffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx - 20, cy);
            ctx.lineTo(cx + 20, cy);
            ctx.moveTo(cx, cy - 20);
            ctx.lineTo(cx, cy + 20);
            ctx.stroke();

            // Draw north indicator
            const orientation = frameData.orientation;
            const northAngle = (orientation - 90) * Math.PI / 180;
            const northLength = 60;
            ctx.strokeStyle = '#ff0000';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(
                cx + Math.cos(northAngle) * northLength,
                cy + Math.sin(northAngle) * northLength
            );
            ctx.stroke();

            ctx.fillStyle = '#ff0000';
            ctx.font = 'bold 14px Courier New';
            ctx.fillText('N',
                cx + Math.cos(northAngle) * (northLength + 15),
                cy + Math.sin(northAngle) * (northLength + 15)
            );

            // Convert stars data and calculate proper scaling
            const stars = frameData.stars;
            
            if (stars.length === 0) return;

            // Convert star data format from [id, x, y] to {{id, x, y}}
            const starsFormatted = stars.map(star => ({{
                id: star[0],
                x: star[1],
                y: star[2]
            }}));

            // Scale based on sensor resolution to canvas size
            const scaleX = canvas.width / SENSOR_WIDTH;
            const scaleY = canvas.height / SENSOR_HEIGHT;

            // Draw connections between nearby stars first (so they appear behind stars)
            ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
            ctx.lineWidth = 1;
            for (let i = 0; i < starsFormatted.length; i++) {{
                for (let j = i + 1; j < starsFormatted.length; j++) {{
                    const x1 = starsFormatted[i].x * scaleX;
                    const y1 = starsFormatted[i].y * scaleY;
                    const x2 = starsFormatted[j].x * scaleX;
                    const y2 = starsFormatted[j].y * scaleY;
                    const dist = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
                    if (dist < 100) {{
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                        ctx.stroke();
                    }}
                }}
            }}

            // Draw stars
            starsFormatted.forEach(star => {{
                const x = star.x * scaleX;
                const y = star.y * scaleY;

                // Create gradient for star glow
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
                gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
                gradient.addColorStop(0.3, 'rgba(0, 255, 255, 0.6)');
                gradient.addColorStop(1, 'rgba(0, 255, 255, 0)');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fill();

                // Draw star center
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, Math.PI * 2);
                ctx.fill();

                // Draw star ID
                ctx.fillStyle = '#00ff00';
                ctx.font = '10px Courier New';
                ctx.fillText(star.id, x + 6, y - 6);
            }});
        }}

        function drawCompass(orientation) {{
            const compass = document.getElementById('compass');
            const size = 80;
            const center = size / 2;
            const angle = orientation;

            const svg = `
                <svg width="${{size}}" height="${{size}}" viewBox="0 0 ${{size}} ${{size}}">
                    <circle cx="${{center}}" cy="${{center}}" r="${{center - 5}}"
                            fill="rgba(0,0,0,0.8)" stroke="#00ff00" stroke-width="2"/>
                    <line x1="${{center}}" y1="${{center}}"
                          x2="${{center + Math.cos((angle - 90) * Math.PI / 180) * (center - 10)}}"
                          y2="${{center + Math.sin((angle - 90) * Math.PI / 180) * (center - 10)}}"
                          stroke="#ff0000" stroke-width="3"/>
                    <text x="${{center}}" y="15" text-anchor="middle" fill="#ff0000"
                          font-family="Courier New" font-size="12" font-weight="bold">N</text>
                    <text x="${{center}}" y="${{size - 5}}" text-anchor="middle" fill="#00ff00"
                          font-family="Courier New" font-size="12">S</text>
                    <text x="8" y="${{center + 5}}" text-anchor="middle" fill="#00ff00"
                          font-family="Courier New" font-size="12">W</text>
                    <text x="${{size - 8}}" y="${{center + 5}}" text-anchor="middle" fill="#00ff00"
                          font-family="Courier New" font-size="12">E</text>
                    <text x="${{center}}" y="${{center + 5}}" text-anchor="middle" fill="#ffff00"
                          font-family="Courier New" font-size="10">${{angle.toFixed(1)}}°</text>
                </svg>
            `;
            compass.innerHTML = svg;
        }}

        function updateInfoPanel(frameData) {{
            document.getElementById('ra-deg').textContent = frameData.ra.toFixed(3) + '°';
            document.getElementById('dec-deg').textContent = frameData.dec.toFixed(3) + '°';
            document.getElementById('scale').textContent = frameData.scale.toFixed(2) + ' arcsec/px';
            document.getElementById('orientation').textContent = frameData.orientation.toFixed(1) + '°';
            document.getElementById('num-stars').textContent = frameData.num_stars;
            document.getElementById('roll').textContent = frameData.orientation.toFixed(1) + '°';
            document.getElementById('pitch').textContent = frameData.dec.toFixed(1) + '°';
            document.getElementById('yaw').textContent = frameData.ra.toFixed(1) + '°';
            document.getElementById('frameCounter').textContent = currentFrame + 1;
        }}

        function drawFrame() {{
            if (framesData.length === 0) return;
            
            const frameData = framesData[currentFrame];
            drawStarField(frameData);
            drawCompass(frameData.orientation);
            updateInfoPanel(frameData);
        }}

        function nextFrame() {{
            currentFrame = (currentFrame + 1) % framesData.length;
            drawFrame();
        }}

        function prevFrame() {{
            currentFrame = (currentFrame - 1 + framesData.length) % framesData.length;
            drawFrame();
        }}

        function play() {{
            if (!isPlaying) {{
                isPlaying = true;
                playInterval = setInterval(nextFrame, 1000);
            }}
        }}

        function pause() {{
            isPlaying = false;
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}

        // Event listeners
        document.getElementById('prevBtn').addEventListener('click', () => {{
            pause();
            prevFrame();
        }});

        document.getElementById('nextBtn').addEventListener('click', () => {{
            pause();
            nextFrame();
        }});

        document.getElementById('playBtn').addEventListener('click', play);
        document.getElementById('pauseBtn').addEventListener('click', pause);

        // Initialize
        if (framesData.length > 0) {{
            drawFrame();
        }} else {{
            document.getElementById('totalFrames').textContent = '0';
            document.getElementById('frameCounter').textContent = '0';
        }}
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML animation saved to: {output_path}")


def main():
    """Main execution function."""
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    csv_path = os.path.join(dataset_dir, 'output', 'results', 'star_tracker_trajectory.csv')
    output_path = os.path.join(dataset_dir, 'star_animation.html')

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    frames_data = load_csv_data(csv_path)

    if frames_data:
        create_html_animation(frames_data, output_path)
        print(f"\nVisualization complete!")
        print(f"Output: {output_path}")
        print(f"Open in browser to view animation")
    else:
        print("No successful frames to visualize")


if __name__ == "__main__":
    main()
