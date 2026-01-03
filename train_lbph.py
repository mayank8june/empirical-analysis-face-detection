from training_logic import FaceTrainer
import sys

if __name__ == '__main__':
    trainer = FaceTrainer('lbph')
    if trainer.train():
        print("Training completed successfully")
    else:
        print("Training failed. Ensure you have enough data.")

