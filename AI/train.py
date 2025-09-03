from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO('yolov8n.pt') 

# Define paths
data_yaml = 'dataset/data.yaml'
project_dir = 'runs/train'  # Directory to save training results
experiment_name = 'bolt_training'  # Name for this training run

# Training parameters
training_params = {
    'data': data_yaml,
    'epochs': 100,  # Number of training epochs
    'batch': 16,    # Batch size
    'imgsz': 640,   # Image size
    'device': 0,    # GPU device (use -1 for CPU)
    'patience': 50, # Early stopping patience
    'project': project_dir,
    'name': experiment_name,
    'exist_ok': True,  # Overwrite existing results
    'optimizer': 'Adam',  # Optimizer
    'lr0': 0.001,     # Initial learning rate
}

# Start training
results = model.train(**training_params)

# Print training results
print("Training completed. Results saved in:", os.path.join(project_dir, experiment_name))

# Optional: Evaluate the model on the validation set
metrics = model.val()
print("Validation metrics:", metrics)

# Optional: Export the model to OpenVINO format
model.export(format='openvino')
print("Model exported to OpenVINO format")