# Traffic-scene-detection-on-Indian-roads-under-adverse-weather-condition-using-real-time-video-enhancement-and-YOLO

An Advanced Driver Assistance System (ADAS) pipeline designed to detect traffic signals in extreme fog and rain, specifically tailored for unstructured Indian road conditions.

## 🚀 Overview
Standard object detection models suffer from "weather blindness" in dense fog and heavy monsoon rain, dropping to an 18% mAP. This project overcomes computational limits by combining a fine-tuned YOLOv8-Nano neural network with a deterministic mathematical image-processing pipeline (CLAHE + OpenCV HSV masking). 

## ✨ Key Features
* **Adverse Weather AI:** YOLOv8-Nano model aggressively optimized using AdamW and extreme HSV augmentations.
* **Deterministic Dehazing:** Real-time CLAHE (Contrast Limited Adaptive Histogram Equalization) pipeline to restore contrast in foggy video feeds.
* **Dynamic Color Verification:** OpenCV-based secondary validation layer to filter out false positives like neon signs or taillights.
* **Live Dashboard:** Interactive Streamlit web app for real-time video processing, displaying raw vs. processed feeds side-by-side with dynamic safety alerts.

## 📊 Performance Metrics
* **Baseline mAP@50 (Clear Weather Bias):** 18.0%
* **Optimized mAP@50 (Adverse Weather):** 47.1% (**+29.1% Jump**)
* **False Positive Reduction:** >90% drop in critical detection errors.
* **Inference Speed:** 20+ FPS on a standard CPU.

## 🛠️ Tech Stack
* **Deep Learning:** Ultralytics YOLOv8
* **Computer Vision:** OpenCV
* **Web Interface:** Streamlit
* **Language:** Python 3.10+

## 📂 Dataset
This project was trained and evaluated using the **India Driving Dataset (IDD)** to ensure the model learned from unstructured, real-world Indian traffic conditions. 

Due to standard Git file size limits and licensing, the raw dataset is not included in this repository. You can download the official dataset directly from the IIIT Hyderabad portal:
🔗 **[Download the India Driving Dataset (IDD) Here](https://idd.insaan.iiit.ac.in/)**

*Note: Once downloaded, extract the images and place them in a folder named `/final_dataset/` in the root directory before running any training scripts.*

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Traffic-Scene-Detection-ADAS.git](https://github.com/YourUsername/Traffic-Scene-Detection-ADAS.git)
   cd Traffic-Scene-Detection-ADAS
