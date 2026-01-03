from training_logic import FaceTrainer
import sys

if __name__ == '__main__':
    trainer = FaceTrainer('eigen')
    if trainer.train():
        print("Training completed successfully")
    else:
        print("Training failed. Ensure you have enough data (at least 1 person).")

