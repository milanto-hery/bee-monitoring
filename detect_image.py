import cv2
import argparse
import os
from ultralytics import YOLO
from IPython.display import display, Image

def detect_image(image_path, weights, output_dir="output"):
    """
    Performs detection on a single image and saves the result.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load model
    model = YOLO(weights)
    
    # Run inference
    results = model(image_path)[0]
    
    # Visualize
    annotated_image = results.plot()
    
    # Save result
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"detected_{base_name}")
    cv2.imwrite(save_path, annotated_image)
    
    print(f"--- Detection complete. Result saved to: {save_path} ---")
    
    # Optionally show detection counts
    if len(results.boxes) > 0:
        print("Detected objects:")
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[class_id]
            print(f"- {name}: {conf:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bee Monitoring Image Detection Script")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Path to model weights")
    parser.add_argument("--output", type=str, default="output", help="Directory to save results")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.source):
        detect_image(args.source, args.weights, args.output)
    else:
        print(f"Error: {args.source} is not a valid file.")
