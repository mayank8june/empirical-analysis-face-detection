from recog_logic import FaceRecognizer
import sys

if __name__ == '__main__':
    recognizer = FaceRecognizer('eigen')
    print("Starting Recognition. Press 'q' to quit.")
    recognizer.recognize()
