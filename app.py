from flask import Flask, Response, render_template, jsonify, redirect, url_for
import cv2
import threading
from emotion_model import detect_emotion
import time

app = Flask(__name__)

# Thread-safe globals
data_lock = threading.Lock()
latest_frame = None
latest_detections = []

# Track overall session emotions
emotion_summary = {"Engaged": 0, "Confused": 0, "Bored/Frustrated": 0}
session_active = False

def capture_and_process_frames():
    global latest_frame, latest_detections, emotion_summary, session_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while session_active:
        success, frame = cap.read()
        if not success:
            break

        detections = detect_emotion(frame)

        # Update emotion summary
        for d in detections:
            emo = d["emotion"]
            if emo in emotion_summary:
                emotion_summary[emo] += 1

        # Draw boxes
        for d in detections:
            x, y, w, h = d["box"]
            color = (0, 255, 0)
            if d["emotion"] == "Bored/Frustrated":
                color = (0, 0, 255)
            elif d["emotion"] == "Confused":
                color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID:{d['id']} {d['emotion']}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update shared data
        with data_lock:
            latest_detections = detections
            _, buffer = cv2.imencode(".jpg", frame)
            latest_frame = buffer.tobytes()

    cap.release()

def generate_video_stream():
    global latest_frame
    while True:
        with data_lock:
            if latest_frame is None:
                continue
            frame_bytes = latest_frame
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("Pro-BotV5.2.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data")
def data():
    global latest_detections
    with data_lock:
        detections_copy = latest_detections
    return jsonify(detections_copy)

@app.route("/start_session")
def start_session():
    global session_active, emotion_summary
    emotion_summary = {"Engaged": 0, "Confused": 0, "Bored/Frustrated": 0}
    session_active = True
    threading.Thread(target=capture_and_process_frames, daemon=True).start()
    return "Session started"

@app.route("/stop_session")
def stop_session():
    global session_active
    session_active = False
    time.sleep(1)
    return redirect(url_for("summary"))

@app.route("/summary")
def summary():
    total = sum(emotion_summary.values())
    if total == 0:
        averages = {k: 0 for k in emotion_summary}
    else:
        averages = {k: round(v / total * 100, 2) for k, v in emotion_summary.items()}
    return render_template("summary.html", averages=averages)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
