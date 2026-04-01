# 🛣️ AI Road Damage Severity Detection System

A high-performance FastAPI web application that detects and classifies road damage (potholes, cracks, etc.) in images and videos using YOLOv8. The system includes a modern dashboard, severity scoring, and historical reporting.

## 🚀 Quick Start (Dockerized)

The easiest way to run the project is using Docker. It is optimized for **CPU-only** performance (perfect for Mac M1/M2/M3).

1. **Build the image:**
   ```bash
   docker build -t road-guardian .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8005:8005 --name road-guardian-app road-guardian
   ```

3. **Access the App:**
   Open [http://localhost:8005](http://localhost:8005) in your browser.

## 🛠️ Features
- **Real-time Detection**: Process both images and videos.
- **Severity Scoring**: Automatically classifies damage as Low, Medium, or High based on relative area.
- **Reporting Dashboard**: Track historical scans and clear reports easily.
- **Fully Optimized**: Minimal Docker footprint and ultra-fast build times.

## 📁 Project Structure
- `main.py`: FastAPI backend and routing.
- `model.py`: YOLOv8 inference and video processing logic.
- `utils.py`: Severity calculation helper.
- `templates/`: Modern, responsive HTML templates.
- `models/`: Location for your `.pt` model weights.

## 💻 Local Development (No Docker)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   python main.py
   ```
