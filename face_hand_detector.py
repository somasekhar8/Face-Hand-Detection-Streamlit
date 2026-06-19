import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def detect_face_hands(
    frame,
    holistic_model,
    show_face=True,
    show_right_hand=True,
    show_left_hand=True
):
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

    return image, face_detected, right_hand_detected, left_hand_detected