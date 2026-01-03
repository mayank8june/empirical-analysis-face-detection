import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognitionResearch")

# Project Directories
BASE_DIR = Path(__file__).parent
FACE_DATA_DIR = BASE_DIR / "face_data"
TRAINED_DATA_DIR = BASE_DIR / "trained_data"
HAARCASCADE_DIR = BASE_DIR / "haarcascades"

# Files
CASCADE_PATH = HAARCASCADE_DIR / "haarcascade_frontalface_default.xml"

# Parameters
RESIZE_FACTOR = 4
FACE_WIDTH, FACE_HEIGHT = 112, 92
NUM_TRAINING_IMAGES = 100
CAPTURE_FREQ_DIV = 5

# Research Metrics
BENCHMARK_WINDOW = 30  # Number of frames to average for performance metrics
THRESHOLD_LBPH = 80
THRESHOLD_EIGEN = 4500
THRESHOLD_FISHER = 500

# Ensure directories exist
for directory in [FACE_DATA_DIR, TRAINED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
