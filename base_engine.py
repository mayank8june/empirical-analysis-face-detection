import cv2
from config import CASCADE_PATH, logger

class FaceEngine:
    def __init__(self):
        if not CASCADE_PATH.exists():
            logger.error(f"Cascade file not found at {CASCADE_PATH}")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
            if self.face_cascade.empty():
                logger.error("Failed to load cascade classifier.")

    def detect_faces(self, gray_img, scaleFactor=1.1, minNeighbors=5):
        if self.face_cascade is None:
            return []
        return self.face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    def get_largest_face(self, faces):
        if len(faces) == 0:
            return None
        areas = [w * h for (x, y, w, h) in faces]
        return faces[areas.index(max(areas))]
