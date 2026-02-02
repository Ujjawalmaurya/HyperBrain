import argparse
from ultralytics import YOLO

# Datasets

# https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset/data/code
# https://www.kaggle.com/datasets/emmarex/plantdisease/data
# https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset

def train(epochs=100, batch=16, model_size='x', device='0'):
    # Load a model
    # yolov8n-cls.pt (nano), yolov8s-cls.pt (small), yolov8m-cls.pt (medium), yolov8l-cls.pt (large), yolov8x-cls.pt (extra large)
    model_name = f'yolov8{model_size}-cls.pt'
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)  # load a pretrained model (recommended for training)

    # Train the model
    # data argument points to the directory containing 'train' and 'val' folders
    results = model.train(
        data='/home/um/Stuffs/HyperSpaceHAckathon/HyperBrain/datasets/unified', 
        epochs=epochs, 
        imgsz=640,
        batch=batch,
        device=device, # Use 0 for first GPU, or 'cpu' if no GPU
        project='hyperbrain_run',
        name=f'yolov8{model_size}_combined',
        exist_ok=True
    )
    
    print("Training Complete. Validation results:")
    metrics = model.val() # validate model
    print(f"Top-1 Accuracy: {metrics.top1}")
    print(f"Top-5 Accuracy: {metrics.top5}")
    
    # Export
    success = model.export(format='onnx')
    print(f"Export Success: {success}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 Classification Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--size', type=str, default='x', choices=['n', 's', 'm', 'l', 'x'], help='Model size (n, s, m, l, x)')
    
    parser.add_argument('--device', type=str, default='0', help='Device to use (0, 0,1, cpu)')
    
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch=args.batch, model_size=args.size, device=args.device)
