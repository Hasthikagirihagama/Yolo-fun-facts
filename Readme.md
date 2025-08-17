# 📦 YOLO Object Detection with Fun Facts (Web + Mobile Optimized)

This project uses **YOLOv8 (Ultralytics)** to detect objects in real time through a web interface, and then shows **fun facts** about the detected objects. Works both on **desktop** and **mobile** (with support for the back camera on phones).

---

## ⚡ Features

* Real-time object detection using **YOLOv8**
* Works on both **desktop & mobile browsers**
* Uses **back camera** by default on mobile
* **Bounding boxes + labels** drawn directly on video
* Fun facts displayed beside the video

---

## 🔧 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/yolo-facts-app.git
cd yolo-facts-app
```

### 2. Create and activate a virtual environment

```bash
# Create env
python -m venv venv

# Activate env
# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install ultralytics flask opencv-python
```

### 4. Run Flask server

```bash
python app.py
```

You should see something like:

```
Running on http://127.0.0.1:5000
```

### 5. Access on browser

* Desktop: Open `http://127.0.0.1:5000`
* Mobile: Use [Ngrok](https://ngrok.com/) or connect both devices to the same WiFi and use your machine’s IP, e.g.:

  ```
  http://192.168.1.5:5000
  ```

---


## 🧪 Example

* Detects an **apple** 🍎 → Shows fun fact like: *“Apples float in water because 25% of their volume is air.”*
* Detects a **dog** 🐶 → Shows: *“Dogs have a sense of smell up to 40x stronger than humans.”*

---
