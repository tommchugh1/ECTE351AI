if __name__ == '__main__':

    from ultralytics import YOLO
    import torch
    import intel_extension_for_pytorch as ipex
    import os

    # Check for XPU
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU not detected.")
    print(f"XPU Detected: {torch.xpu.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"IPEX Version: {ipex.__version__}")

    # Set device
    device = torch.device("xpu:0")

    # Define paths
    header = os.path.dirname(os.path.realpath(__file__))
    data_yaml = 'C:\\Users\\Group8\\Desktop' + '\\' + 'dataset/data.yaml'  # data.yaml directory
    project_dir = header + '\\' + 'RUNS/train'       # Directory to save training results
    experiment_name = 'bolt_training'                # Name for this training run
    model_path = header + '\\' + 'yolov8n.yaml'      # Directory for model to be saved

    # Load yolov8 model
    #retrained = YOLO(model_path)
    retrained = YOLO('yolov8n.yaml')
    retrained.model.to(torch.device("cpu"))
    print("Model loaded on CPU")

    # Verify paths
    #if not os.path.exists(data_yaml):
    #    raise FileNotFoundError("data_yaml file not found.")
    #if not os.path.exists(model_path):
    #    raise FileNotFoundError("Model file not found.")

    # Move model to XPU
    try:
        retrained.model.to(device)
        torch.xpu.empty_cache()
        print("Model moved to XPU")
    except Exception as e:
        print(f"Failed to move model to XPU: {str(e)}")
        device = torch.device("cpu")
        retrained.model.to(device)
        retrained = ipex.optimize(retrained, dtype=torch.float32, inplace=True)
        print("Falling back to CPU")


    # Training parameters
    training_params = {
        'data': data_yaml,
        'epochs': 100,        # Number of training epochs
        'batch': 16,          # Batch size
        'imgsz': 640,         # Image size
        'device': device,
        'patience': 50,       # Early stopping patience
        'project': project_dir,
        'name': experiment_name,
        'exist_ok': True,     # Overwrite existing results
        'optimizer': 'Adam',  # Optimizer
        'lr0': 0.001,         # Initial learning rate
        'amp': True,          # Handled by custom_check_amp
        'cos_lr': True,
    }


    # Start training
    try:
        print(f"Starting training on {device.type.upper()}...")
        if torch.xpu.is_available():
            training_params['device'] = torch.device('xpu:0')
            # Disable AMP to avoid CUDA-related checks
            training_params['amp'] = False
        results = retrained.train(**training_params)
        print("Training completed. Results saved in: %s", os.path.join(project_dir, experiment_name))
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


# Export the trained model to OpenVINO format
    retrained_model_path = os.path.join(header, "RUNS", "train", "bolt_training", "weights", "best.pt")
    model = YOLO(retrained_model_path)
    model.export(format="openvino", imgsz=640)
