import cv2
import numpy as np
import os
from pathlib import Path
from config import FACE_DATA_DIR, TRAINED_DATA_DIR, logger

class FaceTrainer:
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == 'eigen':
            self.model = cv2.face.EigenFaceRecognizer_create()
            self.save_path = TRAINED_DATA_DIR / 'eigen_trained_data.xml'
        elif model_type == 'fisher':
            self.model = cv2.face.FisherFaceRecognizer_create()
            self.save_path = TRAINED_DATA_DIR / 'fisher_trained_data.xml'
        elif model_type == 'lbph':
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.save_path = TRAINED_DATA_DIR / 'lbph_trained_data.xml'
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def load_dataset(self):
        images, labels = [], []
        label_map = {}
        current_id = 0
        
        person_dirs = [d for d in FACE_DATA_DIR.iterdir() if d.is_dir()]
        
        if not person_dirs:
            logger.error("No training data found in face_data directory.")
            return None, None, None

        logger.info(f"Loading dataset for {self.model_type} training...")
        
        for person_dir in person_dirs:
            label_map[current_id] = person_dir.name
            count = 0
            for img_path in person_dir.glob("*.png"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(current_id)
                    count += 1
            logger.info(f"Loaded {count} images for {person_dir.name}")
            current_id += 1
            
        if not images:
            logger.error("No valid images found.")
            return None, None, None
            
        return images, np.array(labels), label_map

    def train(self):
        images, labels, label_map = self.load_dataset()
        
        if images is None or len(images) == 0:
            return False

        if self.model_type == 'fisher' and len(set(labels)) < 2:
            logger.error("FisherFace training requires at least 2 different people.")
            return False

        try:
            logger.info(f"Starting {self.model_type} training...")
            self.model.train(images, labels)
            self.model.save(str(self.save_path))
            logger.info(f"Model saved to {self.save_path}")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
