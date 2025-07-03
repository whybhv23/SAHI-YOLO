# 🔍 SAHI-Style Sliced Inference on Hailo YOLO Model

This project implements **sliced image inference** using **SAHI-like slicing** with a **YOLO model running on the Hailo AI Accelerator**. It is designed to detect small or dense objects in large images by breaking the image into overlapping slices, running inference on each, and combining (stitching) the results.

---

## 📦 Features

- SAHI-style image slicing using `sahi.slicing.slice_image`
- Inference on each patch using Hailo's `.hef` compiled YOLO model
- Bounding box stitching (offset correction) to reconstruct full-image detections
- OpenCV-based result visualization

---

## ⚙️ Requirements

- Python 3.8+
- Hailo instance of picamera2
- opencv
- Python packages:
  ```bash
  pip install numpy opencv-python sahi torch torchvision
