import cv2
import torch
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Cargar el modelo YOLO
model = YOLO("yolov10n.pt")

def detect_objects(frame):
    # Realizar la detección
    results = model(frame)
    
    # Dibujar las detecciones en el frame
    annotated_frame = results[0].plot()
    
    return annotated_frame

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_objects(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)