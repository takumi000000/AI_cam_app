# Colab AI Camera Demo (Face Mesh + Hands)

以下は **Google Colab** で動作するデモ用コードです。セルを上から順に実行してください。

---

## 1. インストール

```bash
!pip install mediapipe opencv-python
```

---

## 2. JavaScript セル（Webカメラ取得）

```javascript
// @title Webcam Access
const video = document.createElement('video');
video.style.width = '640px';
video.style.height = '480px';
video.setAttribute('playsinline', '');
video.setAttribute('autoplay', '');
video.setAttribute('muted', '');

document.body.appendChild(video);

navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  video.srcObject = stream;
});
```

---

## 3. Python セル（MediaPipe + OpenCV デモ）

```python
# @title MediaPipe Face Mesh + Hands Demo
import cv2
import numpy as np
import mediapipe as mp
import time
import base64
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

# JavaScript側のWebカメラを取得
# Colabでカメラ画像をフレーム単位で取得する関数

def js_get_frame():
    js = """
    async function capture() {
      const video = document.querySelector('video');
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.8);
    }
    capture();
    """
    data = eval_js(js)
    return b64decode(data.split(',')[1])

# MediaPipe初期化
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

# 色設定（サイバー風）
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

# 擬似ストレス値
stress_value = 50.0

# メインループ（適度に軽量化）
for _ in range(300):  # 約10秒程度
    frame = js_get_frame()
    img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(img_rgb)
    hands_results = hands.process(img_rgb)

    h, w, _ = img.shape

    # Face Mesh描画
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                img,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=CYAN, thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=GREEN, thickness=1),
            )

            # 簡易ストレス推定（眉間の距離）
            lm = face_landmarks.landmark
            brow_left = lm[70]
            brow_right = lm[300]
            dist = np.linalg.norm(
                np.array([brow_left.x - brow_right.x, brow_left.y - brow_right.y])
            )
            stress_value = 100 - (dist * 300)  # 適当にスケール
            stress_value = np.clip(stress_value + np.random.uniform(-5, 5), 0, 100)

    # Hands描画
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=MAGENTA, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=GREEN, thickness=2),
            )

    # HUD風オーバーレイ
    cv2.putText(img, f"Stress Level: {stress_value:.1f}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, CYAN, 2)

    # System Log表示
    log_y = 80
    cv2.putText(img, "System Log:", (20, log_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    log_y += 25

    # 一部ランドマーク座標を表示
    if face_results.multi_face_landmarks:
        lm0 = face_results.multi_face_landmarks[0].landmark[1]
        cv2.putText(img, f"Face LM[1]: {lm0.x:.2f}, {lm0.y:.2f}", (20, log_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1)
        log_y += 20

    if hands_results.multi_hand_landmarks:
        lm1 = hands_results.multi_hand_landmarks[0].landmark[0]
        cv2.putText(img, f"Hand LM[0]: {lm1.x:.2f}, {lm1.y:.2f}", (20, log_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1)

    # 表示
    _, jpg = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(jpg.tobytes()).decode('utf-8')
    display(Javascript('''
        var img = document.getElementById('output');
        if (!img) {
            img = document.createElement('img');
            img.id = 'output';
            document.body.appendChild(img);
        }
        img.src = 'data:image/jpg;base64,%s';
    ''' % (b64,)))

    time.sleep(0.03)
```
