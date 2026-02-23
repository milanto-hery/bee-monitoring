import argparse
import os
from ultralytics import YOLO
from roboflow import Roboflow

def train_model(data_yaml, epochs, model_size, imgsz=640):
    """
    Trains the YOLO model with specified parameters.
    """
    model_name = f"yolo11{model_size}.pt"
    print(f"--- Initializing training with {model_name} ---")
    
    # Load a model
    model = YOLO(model_name)
    
    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        plots=True,
        patience=20
    )
    print("--- Training completed ---")

def download_dataset(api_key, workspace, project_id, version):
    """
    Downloads the dataset from Roboflow.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)
    dataset = project.version(version).download("yolov8")
    return os.path.join(dataset.location, "data.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bee Monitoring Training Script")
    
    # Dataset arguments
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--roboflow_key", type=str, help="Roboflow API Key")
    parser.add_argument("--workspace", type=str, default="project_name", help="Roboflow workspace name")
    parser.add_argument("--project", type=str, default="project_id", help="Roboflow project ID")
    parser.add_argument("--version", type=int, default=1, help="Roboflow dataset version")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model_size", type=str, default="n", choices=['n', 's', 'm', 'l', 'x'], help="YOLO model size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")

    args = parser.parse_args()

    # Determine data.yaml path
    if args.data:
        data_path = args.data
    elif args.roboflow_key:
        data_path = download_dataset(args.roboflow_key, args.workspace, args.project, args.version)
    else:
        print("Error: Either --data or --roboflow_key must be provided.")
        exit(1)

    train_model(data_path, args.epochs, args.model_size, args.imgsz)
