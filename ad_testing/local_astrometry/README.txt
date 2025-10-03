Star Dataset Astrometry with Astroquery (Local Processing)

This Docker project runs star mapping and astrometry locally using astroquery.

BUILD:
  docker build -t star-astrometry .

RUN:
  docker run -v "%cd%\dataset:/app/dataset" star-astrometry

The dataset folder will be mounted so results are saved to your local machine.

Files:
  - Dockerfile: Container configuration
  - main_local.py: Main script using astroquery
  - requirements.txt: Python dependencies
  - .dockerignore: Files to exclude from build
