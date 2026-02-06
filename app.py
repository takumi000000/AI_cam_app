import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template_string

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

HTML = """
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8" />
    <title>AI Camera Demo</title>
    <style>
      body { background:#0b0f14; color:#d9f7ff; font-family: sans-serif; }
      .wrap { display:flex; flex-direction:column; align-items:center; }
      img { border:2px solid #00f5ff; box-shadow:0 0 20px #00f5ff55; }
      .note { margin-top:12px; opacity:0.7; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>AI Camera Demo</h1>
      <img src="/video_feed" width="640" height="480" />
      <div class="note">/video_feed をMJPEGストリームで配信中</div>
    </div>
  </body>
</html>
"""


def draw_overlay(frame, face_results, hands_results, stress_value):
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=CYAN, thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=GREEN, thickness=1),
            )

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=MAGENTA, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=GREEN, thickness=2),
            )

    cv2.putText(
        frame,
        f"Stress Level: {stress_value:.1f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        CYAN,
        2,
    )

    log_y = 80
    cv2.putText(
        frame,
        "System Log:",
        (20, log_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        GREEN,
        2,
    )
    log_y += 25

    if face_results.multi_face_landmarks:
        lm0 = face_results.multi_face_landmarks[0].landmark[1]
        cv2.putText(
            frame,
            f"Face LM[1]: {lm0.x:.2f}, {lm0.y:.2f}",
            (20, log_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            CYAN,
            1,
        )
        log_y += 20

    if hands_results.multi_hand_landmarks:
        lm1 = hands_results.multi_hand_landmarks[0].landmark[0]
        cv2.putText(
            frame,
            f"Hand LM[0]: {lm1.x:.2f}, {lm1.y:.2f}",
            (20, log_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            CYAN,
            1,
        )


@app.route("/")
def index():
    return render_template_string(HTML)


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Webcam failed to open. Check /dev/video0 access.")

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    stress_value = 50.0
    recent_dist = deque(maxlen=5)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            hands_results = hands.process(rgb)

            if face_results.multi_face_landmarks:
                lm = face_results.multi_face_landmarks[0].landmark
                brow_left = lm[70]
                brow_right = lm[300]
                dist = np.linalg.norm(
                    np.array([brow_left.x - brow_right.x, brow_left.y - brow_right.y])
                )
                recent_dist.append(dist)
                avg_dist = float(np.mean(recent_dist)) if recent_dist else dist
                stress_value = 100 - (avg_dist * 300)
                stress_value = np.clip(
                    stress_value + np.random.uniform(-3, 3), 0, 100
                )

            draw_overlay(frame, face_results, hands_results, stress_value)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()
        face_mesh.close()
        hands.close()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
