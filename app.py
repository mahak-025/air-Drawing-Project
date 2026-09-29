from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", max_http_buffer_size=5_000_000)

# ---------------- MODEL DOWNLOAD ----------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

# ---------------- STATE (single-user demo) ----------------
W, H = 640, 480
canvas = np.zeros((H, W, 3), dtype=np.uint8)
draw_color = (92, 92, 255)   # BGR
brush_thickness = 6
prev_x, prev_y = None, None


def extended_fingers(landmarks):
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return sum(1 for tip, pip in pairs if landmarks[tip].y < landmarks[pip].y)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/set_color", methods=["POST"])
def set_color():
    global draw_color
    hex_color = request.json["color"].lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    draw_color = (b, g, r)
    return jsonify({"status": "ok"})


@app.route("/set_brush", methods=["POST"])
def set_brush():
    global brush_thickness
    brush_thickness = int(request.json["size"])
    return jsonify({"status": "ok"})


@app.route("/clear", methods=["POST"])
def clear_canvas():
    global canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    return jsonify({"status": "ok"})


# ---------------- SOCKET: receives frames from browser webcam ----------------
@socketio.on("frame")
def handle_frame(data):
    global prev_x, prev_y, canvas

    # data is a base64 data URL: "data:image/jpeg;base64,...."
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (W, H))
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)

    mode_text = "Idle"

    if results.hand_landmarks:
        landmarks = results.hand_landmarks[0]
        count = extended_fingers(landmarks)
        index_tip = landmarks[8]
        x = int(index_tip.x * W)
        y = int(index_tip.y * H)

        if count >= 4:
            mode_text = "Eraser"
            cv2.circle(canvas, (x, y), 35, (0, 0, 0), -1)
            cv2.circle(frame, (x, y), 35, (255, 255, 255), 2)
            prev_x, prev_y = None, None
        elif count == 1:
            mode_text = "Drawing"
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)
            prev_x, prev_y = x, y
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        else:
            prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    combined = cv2.addWeighted(frame, 0.6, canvas, 0.7, 0)
    cv2.putText(combined, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    _, buffer = cv2.imencode(".jpg", combined)
    result_b64 = base64.b64encode(buffer).decode("utf-8")
    socketio.emit("processed_frame", "data:image/jpeg;base64," + result_b64)



import os
if __name__ == "__main__":
     port = int(os.environ.get("PORT", 5000)) 
     socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)