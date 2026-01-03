import cv2
import numpy as np
import os
from pathlib import Path
from config import *
from base_engine import FaceEngine, logger

import time
from collections import deque

class FaceRecognizer(FaceEngine):
    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        if model_type == 'eigen':
            self.model = cv2.face.EigenFaceRecognizer_create()
            self.model_path = TRAINED_DATA_DIR / 'eigen_trained_data.xml'
            self.threshold = THRESHOLD_EIGEN
        elif model_type == 'fisher':
            self.model = cv2.face.FisherFaceRecognizer_create()
            self.model_path = TRAINED_DATA_DIR / 'fisher_trained_data.xml'
            self.threshold = THRESHOLD_FISHER
        elif model_type == 'lbph':
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.model_path = TRAINED_DATA_DIR / 'lbph_trained_data.xml'
            self.threshold = THRESHOLD_LBPH
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.label_map = {}
        self.load_model()
        
        # Metrics tracking
        self.inference_times = deque(maxlen=BENCHMARK_WINDOW)
        self.fps_deque = deque(maxlen=BENCHMARK_WINDOW)
        self.last_time = time.time()

    def load_model(self):
        if not self.model_path.exists():
            logger.warning(f"Model file {self.model_path} not found. Please train first.")
            return False

        try:
            self.model.read(str(self.model_path))
            person_dirs = sorted([d for d in FACE_DATA_DIR.iterdir() if d.is_dir()])
            for i, person_dir in enumerate(person_dirs):
                self.label_map[i] = person_dir.name
            logger.info(f"Loaded {self.model_type} model and {len(self.label_map)} labels.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def recognize(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logger.error("Could not open video device")
            return

        print(f"Starting {self.model_type} Research Evaluation. Press 'q' to exit.")

        while True:
            ret, frame = video_capture.read()
            if not ret or frame is None:
                continue

            start_time = time.time()
            processed_frame = self.process_frame(frame)
            
            # Calculate metrics
            curr_time = time.time()
            dt = curr_time - self.last_time
            if dt > 0:
                self.fps_deque.append(1.0 / dt)
            self.last_time = curr_time

            # Add overlay info
            avg_fps = sum(self.fps_deque) / len(self.fps_deque) if self.fps_deque else 0
            avg_inf = (sum(self.inference_times) / len(self.inference_times)) * 1000 if self.inference_times else 0
            
            cv2.putText(processed_frame, f"Algorithm: {self.model_type.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Inference Time: {avg_inf:.2f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(processed_frame, f"FPS: {avg_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(f'Research Evaluation - {self.model_type.upper()}', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        display_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        
        small_gray = cv2.resize(gray, (gray.shape[1] // RESIZE_FACTOR, gray.shape[0] // RESIZE_FACTOR))
        faces = self.detect_faces(small_gray)
        
        for face in faces:
            x, y, w, h = [v * RESIZE_FACTOR for v in face]
            face_roi = gray[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                face_resized = cv2.resize(face_roi, (FACE_WIDTH, FACE_HEIGHT))
                
                inf_start = time.time()
                try:
                    label_id, confidence = self.model.predict(face_resized)
                    self.inference_times.append(time.time() - inf_start)
                    
                    is_known = confidence < self.threshold
                    
                    if is_known and label_id in self.label_map:
                        name = self.label_map[label_id]
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame, f"{name}: Dist={int(confidence)}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    logger.error(f"Prediction error: {e}")

        return display_frame
