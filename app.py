from ultralytics import YOLO
import cv2
import google.generativeai as genai
import time

# Setup Gemini
genai.configure(api_key="AIzaSyDrTNY5e-PUD2Il7J9XNiCOxTNGUPM-DWc")
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# Load YOLOv8 Nano
model_yolo = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# Track last call times per label for cooldown
last_call_time = {}
cooldown_seconds = 5  # wait 3 seconds before re-calling Gemini for the same object

facts_cache = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model_yolo.predict(frame, verbose=False)
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detected_label = model_yolo.names[cls_id]
            now = time.time()

            # Check cooldown timer and cache
            if (detected_label not in last_call_time) or (now - last_call_time[detected_label] > cooldown_seconds):
                
                # Small delay for stability
                time.sleep(0.3)

                # Only call Gemini if not cached
                if detected_label not in facts_cache:
                    prompt = f"Give me 3 short, fun facts about a {detected_label} in under 15 words each."
                    response = model_gemini.generate_content(prompt)
                    # Split facts by newline and strip empty lines
                    facts_cache[detected_label] = [fact.strip("- ").strip() for fact in response.text.strip().split("\n") if fact.strip()]
                
                print(f"Facts about {detected_label}:")
                for fact in facts_cache[detected_label]:
                    print("-", fact)

                last_call_time[detected_label] = now

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
