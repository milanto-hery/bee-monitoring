# 🐝 Bee Monitoring

This system identifies, tracks, and logs the activity of workers, pollen foragers, drones, and queens using deep learning. Monitoring beehives is essential for assessing colony health and pollination productivity. This project provides a non-invasive tool to track hive traffic and detect rare events (like queen movement) without disturbing the bees.

---

## Core Objectives

- Real-time Counting: Quantify hive traffic (In/Out) and foraging intensity.
- Classification: Distinguish between workers, pollen-bearing bees, drones, and queens.
- Behavioral Logging: Record exact timestamps of significant biological events.

## Key Features

`Unique Tracking & Anti-Double Counting`. Utilizing ByteTrack, the system assigns a unique ID to every bee. This ensures that a bee lingering at the entrance is only counted once, providing accurate demographic data.

`Transparent Analytics Dashboard`. The video output features a semi-transparent bottom overlay (HUD) that displays:

- Live Stats: Current count of each bee class.Suspicious
- Queen Events: A log of potential queen sightings.
- System Time: Real-time device clock integration for precise logging.
  
`False Positive Mitigation (Queen Verification)`. To minimize errors where a large worker might be mistaken for a queen, the system employs:

- Aspect Ratio Analysis: Checks if the body shape matches the long, slender "prototype" of a queen.
- Persistence Filtering: A detection must be consistent across multiple frames (15+) before being logged.
  
## Training

The model was trained on a robust dataset from Roboflow, featuring thousands of annotated images to ensure the AI recognizes bees from various angles and lighting conditions.
- Validation & Testing: Initial verification using the Test Split (unseen data) from the training set.
- Dynamic Testing: Real-world performance validation using high-quality hive entrance footage from YouTube.

---

## Installation & Usage
### Clone the Repository:
    
      git clone https://github.com/milanto-hery/bee-monitoring.git
  
### Install Dependencies:

      pip install ultralytics opencv-python numpy
  
###  Run the Notebook:
      
      Open bee_monitoring.ipynb and follow the cells to process your video files.
      
---

## 📊 Sample Output Representation
This section showcases the model performing real-time object detection on a video stream.


<p align="left"> <img src="./output/bee_monitoring_demo.gif" width="1000px"> </p>

---

## 📜 Ethical Considerations & Acknowledgments
This project is intended for research and conservation purposes. It promotes non-invasive monitoring to protect honey bee populations worldwide.
- **Data Credits**: Roboflow for the training dataset.
- **Video Credits**: YouTube Beehive Livestreams used for testing purposes. [Cherylee's Bees] /https://www.https://www.youtube.com/watch?v=u5X0ymHfY9Y

---
