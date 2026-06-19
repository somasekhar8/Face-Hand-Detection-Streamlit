import streamlit as st
import cv2
import time
import mediapipe as mp

st.set_page_config(
    page_title="Face & Hand Detection",
    page_icon="🖐️",
    layout="wide"
)

# ---------------- CSS Styling ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00E5FF;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #CCCCCC;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #333333;
    text-align: center;
    margin-bottom: 15px;
}
.metric-text {
    color: #00FF88;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Face & Hand Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Real-time Face Mesh and Hand Landmark Detection using OpenCV, MediaPipe and Streamlit</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Controls")

if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

if st.sidebar.button("▶️ Start Camera"):
    st.session_state.run_camera = True

if st.sidebar.button("⏹️ Stop Camera"):
    st.session_state.run_camera = False

show_face = st.sidebar.checkbox("Show Face Landmarks", value=True)
show_right_hand = st.sidebar.checkbox("Show Right Hand", value=True)
show_left_hand = st.sidebar.checkbox("Show Left Hand", value=True)

detection_confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1, 1.0, 0.5
)

tracking_confidence = st.sidebar.slider(
    "Tracking Confidence",
    0.1, 1.0, 0.5
)

st.sidebar.info("Use Start Camera and Stop Camera buttons. Press Ctrl + C in terminal to stop server.")

# ---------------- Layout ----------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📷 Live Camera Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Detection Status")
    fps_placeholder = st.empty()
    face_status = st.empty()
    right_hand_status = st.empty()
    left_hand_status = st.empty()

# ---------------- MediaPipe Setup ----------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence
)

cap = cv2.VideoCapture(0)
previousTime = time.time()

# ---------------- Main Logic ----------------
while st.session_state.run_camera:
    ret, frame = cap.read()

    if not ret:
        st.error("Camera not detected. Please check webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (900, 650))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_detected = results.face_landmarks is not None
    right_hand_detected = results.right_hand_landmarks is not None
    left_hand_detected = results.left_hand_landmarks is not None

    if show_face and face_detected:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        )

    if show_right_hand and right_hand_detected:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

    if show_left_hand and left_hand_detected:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(
        image,
        f"FPS: {int(fps)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(image, channels="RGB", use_container_width=True)

    fps_placeholder.markdown(
        f'<div class="card"><p>FPS</p><div class="metric-text">{int(fps)}</div></div>',
        unsafe_allow_html=True
    )

    face_status.markdown(
        f'<div class="card"><p>Face</p><div class="metric-text">{"Detected ✅" if face_detected else "Not Detected ❌"}</div></div>',
        unsafe_allow_html=True
    )

    right_hand_status.markdown(
        f'<div class="card"><p>Right Hand</p><div class="metric-text">{"Detected ✅" if right_hand_detected else "Not Detected ❌"}</div></div>',
        unsafe_allow_html=True
    )

    left_hand_status.markdown(
        f'<div class="card"><p>Left Hand</p><div class="metric-text">{"Detected ✅" if left_hand_detected else "Not Detected ❌"}</div></div>',
        unsafe_allow_html=True
    )

cap.release()
holistic_model.close()