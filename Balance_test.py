import cv2
import mediapipe as mp
import time
import platform
from datetime import datetime

# ========================
# Config
# ========================
NORMAL_MAX = 1.0      # 0–1s -> Normal
FATIGUE_MAX = 3.0     # >1–3s -> Fatigued
TIMEOUT = 12.0        # No movement after 12s -> No movement (flag)
WINDOW_TITLE = "Morpheus EVA Fatigue Check"
PREFERRED_INDEXES = (0, 1, 2, 3)

# ========================
# Cross-platform camera open
# ========================
def open_camera(indexes=PREFERRED_INDEXES):
    sys = platform.system()
    # Choose preferred backend per OS
    if sys == "Darwin":      # macOS
        backends = [cv2.CAP_AVFOUNDATION, None]
    elif sys == "Windows":   # Windows
        backends = [cv2.CAP_DSHOW, None]
    else:                    # Linux / other
        backends = [cv2.CAP_V4L2, None]

    for i in indexes:
        for be in backends:
            cap = cv2.VideoCapture(i) if be is None else cv2.VideoCapture(i, be)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"[DEBUG] Using camera index {i} on {sys} "
                          f"backend={'default' if be is None else be}")
                    return cap
            cap.release()
    raise RuntimeError("No usable camera found. Close other apps or try different indexes.")

# ========================
# Helpers
# ========================
def classify_delay(d):
    if d <= NORMAL_MAX:
        return f"✅ Normal response ({d:.2f}s)", 2
    elif d <= FATIGUE_MAX:
        return f"⚠️ Fatigued response ({d:.2f}s)", 1
    else:
        return f"❗ Problem detected ({d:.2f}s)", 0

def draw_legend(frame):
    h, w = frame.shape[:2]
    x0, y0 = w - 300, 30
    cv2.rectangle(frame, (x0 - 10, y0 - 10), (w - 20, y0 + 140), (0, 0, 0), -1)
    cv2.putText(frame, "Legend:", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"0–{NORMAL_MAX:.0f}s: Normal", (x0, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{NORMAL_MAX:.0f}–{FATIGUE_MAX:.0f}s: Fatigued", (x0, y0 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f">{FATIGUE_MAX:.0f}s: Problem", (x0, y0 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Timeout: {TIMEOUT:.0f}s", (x0, y0 + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

# ========================
# MediaPipe Pose
# ========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ========================
# Start camera
# ========================
cap = open_camera()

# ========================
# State
# ========================
start_time = None
movement_detected = False
delay = 0.0
score = 0
message = "🟡 Raise your right arm above your shoulder"

# ========================
# Main loop
# ========================
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    h, w = frame.shape[:2]

    draw_legend(frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        rs = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        wrist = (int(rw.x * w), int(rw.y * h))
        shoulder = (int(rs.x * w), int(rs.y * h))
        cv2.circle(frame, wrist, 10, (0, 0, 255), -1)
        cv2.circle(frame, shoulder, 10, (255, 0, 0), -1)

        cv2.putText(frame, f"Wrist Y: {rw.y:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Shoulder Y: {rs.y:.2f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if start_time is None and not movement_detected:
            start_time = time.time()
            print("[DEBUG] Timer started")

        # Detect: wrist above shoulder
        if (rw.y < rs.y) and (not movement_detected) and (start_time is not None):
            delay = time.time() - start_time
            movement_detected = True
            message, score = classify_delay(delay)
            print(f"[DEBUG] Movement detected in {delay:.2f} seconds. Score: {score}")

        # Timer & timeout (before movement)
        if (start_time is not None) and (not movement_detected):
            elapsed = time.time() - start_time
            cv2.putText(frame, f"⏱ {elapsed:.2f}s", (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            if elapsed > TIMEOUT:
                message = "❗ No movement detected - Flag for check"
                movement_detected = True
                delay = 0.0
                score = 0
                print(f"[DEBUG] No movement detected within {TIMEOUT}s.")
    else:
        message = "❗ Move back so I can see your full upper body"

    # Status banner
    cv2.putText(frame, message, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if movement_detected else (0, 0, 255), 2)

    if movement_detected:
        cv2.putText(frame, f"Movement Score: {score}/2", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(WINDOW_TITLE, frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

# ========================
# Final result
# ========================
print("\n==== EVA MOVEMENT TEST RESULT ====")
print(f"Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Delay: {delay:.2f}s")
print(f"Score: {score}/2")
if score == 2:
    print("✅ Normal response (0–1s).")
elif score == 1:
    print("⚠️ Potential fatigue-related response (>1–3s).")
else:
    if delay > 0:
        print("❗ Problem detected (>3s).")
    else:
        print("❗ No movement — check condition.")

cap.release()
cv2.destroyAllWindows()

