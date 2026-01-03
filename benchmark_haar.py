import cv2
import time
import numpy as np
from collections import deque
from config import CASCADE_PATH, RESIZE_FACTOR

def run_haar_benchmark():
    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open camera")
        return

    fps_stats = deque(maxlen=30)
    det_times = deque(maxlen=30)
    last_time = time.time()

    print("Starting HAAR CASCADE Detection Benchmark. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        display_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        
        # Benchmarking Detection
        start_det = time.time()
        
        # Performance trick: resize for detection
        small_gray = cv2.resize(gray, (0,0), fx=1/RESIZE_FACTOR, fy=1/RESIZE_FACTOR)
        faces = face_cascade.detectMultiScale(small_gray, 1.1, 5)
        
        det_duration = time.time() - start_det
        det_times.append(det_duration)

        # FPS calculation
        curr_time = time.time()
        fps_stats.append(1.0 / (curr_time - last_time))
        last_time = curr_time

        # Draw results
        for (x, y, w, h) in faces:
            x, y, w, h = [v * RESIZE_FACTOR for v in [x, y, w, h]]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Metrics Overlay
        avg_fps = sum(fps_stats) / len(fps_stats) if fps_stats else 0
        avg_det = (sum(det_times) / len(det_times)) * 1000 if det_times else 0
        
        cv2.rectangle(display_frame, (10, 10), (350, 110), (0,0,0), -1)
        cv2.putText(display_frame, "ALGO: HAAR CASCADE (DETECTION)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Det. Latency: {avg_det:.2f} ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"System FPS: {avg_fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Detection Benchmark - HAAR", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_haar_benchmark()
