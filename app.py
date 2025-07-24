from flask import Flask, render_template, Response, jsonify
import cv2
from facial_emotion_recognition import EmotionRecognition
import threading

app = Flask(__name__)
er = EmotionRecognition(device="cpu")
camera = cv2.VideoCapture(0)
is_running = False
lock = threading.Lock()

def generate_frames():
    global is_running
    while True:
        with lock:
            if not is_running:
                continue
        success, frame = camera.read()
        if not success:
            break
        frame = er.recognise_emotion(frame, return_type="BGR")
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global is_running
    with lock:
        is_running = True
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop():
    global is_running
    with lock:
        is_running = False
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True)
