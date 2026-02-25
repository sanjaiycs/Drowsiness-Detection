import cv2
import mediapipe as mp
import numpy as np
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
EAR_THRESHOLD = 0.23
FRAMES_THRESHOLD = 50
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def compute_EAR(landmarks, eye_indices):
    pts = [landmarks[i] for i in eye_indices]

    def dist(a, b):
        return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))

    v1 = dist(pts[1], pts[5])
    v2 = dist(pts[2], pts[4])
    h  = dist(pts[0], pts[3])

    return (v1 + v2) / (2.0 * h)

def main(input_video, output_video):

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    frame_timestamp = 0
    drowsy_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(mp_image, frame_timestamp)
        frame_timestamp += int(1000 / fps)

        status = "Awake"

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear  = compute_EAR(landmarks, LEFT_EYE)
            right_ear = compute_EAR(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            if drowsy_counter > FRAMES_THRESHOLD:
                status = "DROWSY!"

            cv2.putText(frame, f"EAR: {ear:.3f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        color = (0, 255, 0) if status == "Awake" else (0, 0, 255)
        cv2.putText(frame, status, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Output saved as: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mediapipe_drowsiness.py input.mp4 output.mp4")
    else:
        main(sys.argv[1], sys.argv[2])
