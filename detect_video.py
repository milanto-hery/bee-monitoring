import cv2
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
from datetime import datetime

class BeeMonitor:
    def __init__(self, model_path):
        """
        Initializes the BeeMonitor with a YOLO model and ByteTrack tracker.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
        # Define tracking lines for counting (simplified example, can be adjusted)
        # These points should ideally be relative to frame size or provided via CLI
        self.line_start = sv.Point(300, 200)
        self.line_end = sv.Point(300, 500)
        self.line_zone = sv.LineZone(start=self.line_start, end=self.line_end)
        
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.line_zone_annotator = sv.LineZoneAnnotator()
        
        self.event_logs = []

    def log_event(self, message):
        """
        Logs an event to the internal list and a text file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.event_logs.append(log_entry)
        
        with open("queen_events.txt", "a") as f:
            f.write(log_entry + "\n")
        
        if len(self.event_logs) > 5:
            self.event_logs.pop(0)

    def process_frame(self, frame):
        """
        Processes a single frame for detection, tracking, and counting.
        """
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        # Update counting
        self.line_zone.trigger(detections)

        # Check for Queen (class_id might vary based on dataset, usually Queen is unique)
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            class_name = self.model.names[class_id]
            
            if class_name.lower() == "queen" and confidence > 0.5:
                self.log_event("Suspicious Queen detected!")

        # Annotate
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.line_zone_annotator.annotate(state=self.line_zone, scene=annotated_frame)

        return self.draw_hud(annotated_frame)

    def draw_hud(self, frame):
        """
        Draws the red and white 'Bee Monitor AI' overlay and logs.
        """
        h, w = frame.shape[:2]
        now = datetime.now().strftime("%H:%M:%S")
        
        # HUD background
        cv2.rectangle(frame, (0, h - 120), (w, h), (0, 0, 150), -1)  # Reddish footer
        cv2.line(frame, (0, h - 120), (w, h - 120), (255, 255, 255), 2)
        
        # Counts
        cv2.putText(frame, f"IN: {self.line_zone.in_count}", (20, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"OUT: {self.line_zone.out_count}", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Logs
        for i, log in enumerate(self.event_logs):
            cv2.putText(frame, f"> {log}", (200, h - 100 + (i * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Logo and Time
        cv2.putText(frame, "Bee Monitor AI", (w - 220, h - 80),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, now, (w - 180, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return frame

def main():
    parser = argparse.ArgumentParser(description="Bee Monitoring Inference Script")
    parser.add_argument("--source", type=str, required=True, help="Path to video or webcam index")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Path to model weights")
    parser.add_argument("--show", action="store_true", default=True, help="Display video playback")
    parser.add_argument("--output", type=str, help="Path to save processed video")
    
    args = parser.parse_args()

    monitor = BeeMonitor(args.weights)
    
    # Handle webcam or file
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source {args.source}")
        return

    # Video Writer setup if output path is provided
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("--- Starting Bee Monitoring ---")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = monitor.process_frame(frame)
        
        if writer:
            writer.write(processed_frame)
            
        if args.show:
            cv2.imshow("Bee Monitor AI", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("--- Monitoring Stopped ---")

if __name__ == "__main__":
    main()
