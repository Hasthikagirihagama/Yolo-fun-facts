from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Initialize video capture and YOLO model
cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection
        results = model.predict(frame, verbose=False)
        
        # Draw boxes and labels on frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in HTTP multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <body>
    <h1>YOLOv8 Webcam Stream</h1>
    <img src="/video_feed" width="640" height="480" />
    </body>
    </html>
    """

if __name__ == '__main__':
    # Run on all IPs to allow mobile access on same network
    app.run(host='0.0.0.0', port=5000, debug=False)
