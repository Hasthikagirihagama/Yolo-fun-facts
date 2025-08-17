from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import base64
import numpy as np
import google.generativeai as genai
import time

app = Flask(__name__)

# Setup Gemini
genai.configure(api_key="AIzaSyDrTNY5e-PUD2Il7J9XNiCOxTNGUPM-DWc")
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# Load YOLO model
model_yolo = YOLO("yolov8n.pt")

# Cache facts and cooldown tracking
facts_cache = {}
last_call_time = {}
cooldown_seconds = 5

def get_facts(label):
    now = time.time()
    if label in facts_cache and (now - last_call_time[label] < cooldown_seconds):
        return facts_cache[label]

    prompt = f"Give me 3 short, fun facts about a {label} in under 15 words each."
    response = model_gemini.generate_content(prompt)
    facts = [line.strip("- ").strip() for line in response.text.strip().split("\n") if line.strip()]
    facts_cache[label] = facts
    last_call_time[label] = now
    return facts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    img_data = data['image'].split(',')[1]  # remove data:image/jpeg;base64,
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    np_img = np.array(img)

    results = model_yolo.predict(np_img, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model_yolo.names[cls_id]

            facts = get_facts(label)

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'label': label,
                'facts': facts
            })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
