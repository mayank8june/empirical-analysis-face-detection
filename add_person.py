import cv2
import numpy as np
import sys
import os
from pathlib import Path
from config import *
from base_engine import FaceEngine, logger

class PersonCollector(FaceEngine):
    def __init__(self, person_name):
        super().__init__()
        if not person_name:
            raise ValueError("Person name cannot be empty")
        
        self.person_name = person_name
        self.person_dir = FACE_DATA_DIR / person_name
        self.person_dir.mkdir(parents=True, exist_ok=True)
        
        self.count_captures = 0
        self.count_timer = 0

    def collect(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logger.error("Could not open video device")
            return

        logger.info(f"Starting capture for {self.person_name}. Need {NUM_TRAINING_IMAGES} images.")
        
        while self.count_captures < NUM_TRAINING_IMAGES:
            ret, frame = video_capture.read()
            if not ret or frame is None:
                logger.warning("Failed to grab frame")
                continue

            self.count_timer += 1
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('Face Collection - Press Q to Cancel', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Collection cancelled by user.")
                break

        video_capture.release()
        cv2.destroyAllWindows()
        logger.info(f"Finished collection. Captured {self.count_captures} images.")

    def process_frame(self, frame):
        # Mirror the frame for easier positioning
        display_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces on a smaller image for speed
        small_gray = cv2.resize(gray, (gray.shape[1] // RESIZE_FACTOR, gray.shape[0] // RESIZE_FACTOR))
        faces = self.detect_faces(small_gray)
        
        if len(faces) > 0:
            face_sel = self.get_largest_face(faces)
            
            # Scale back to original size
            x, y, w, h = [v * RESIZE_FACTOR for v in face_sel]
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                face_resized = cv2.resize(face_roi, (FACE_WIDTH, FACE_HEIGHT))
                
                if self.count_timer % CAPTURE_FREQ_DIV == 0:
                    self.count_captures += 1
                    img_path = self.person_dir / f"{self.count_captures}.png"
                    cv2.imwrite(str(img_path), face_resized)
                    logger.info(f"Captured {self.count_captures}/{NUM_TRAINING_IMAGES}")

                # Draw feedback
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                progress = int((self.count_captures / NUM_TRAINING_IMAGES) * 100)
                cv2.putText(display_frame, f"{self.person_name}: {progress}%", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_frame

if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    collector = PersonCollector(name)
    collector.collect()
