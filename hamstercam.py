import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# -------------------- File Paths --------------------
current_dir = os.path.dirname(__file__)
awkward_video_path = os.path.join(current_dir, 'hamsterawkward.jpg')
tongue_gif_path = os.path.join(current_dir, 'hamstertongue.gif')

# -------------------- Load awkward hamster frame (from .webm) --------------------
cap_awk = cv2.VideoCapture(awkward_video_path)
ret, frame_awk = cap_awk.read()
if not ret:
    raise Exception("Cannot read hamster video")
cap_awk.release()
awkward_hamster = frame_awk  # BGR image

# -------------------- Load tongue GIF as frames --------------------
gif = Image.open(tongue_gif_path)
tongue_frames = []
try:
    while True:
        frame = np.array(gif.convert("RGBA"))
        tongue_frames.append(frame)
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

print(f"Loaded {len(tongue_frames)} tongue frames")

# -------------------- Mediapipe Face Mesh --------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# -------------------- Open webcam --------------------
cap = cv2.VideoCapture(0)

# -------------------- Overlay function --------------------
def overlay_image(bg, fg):
    h, w = fg.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    if (h != bg_h) or (w != bg_w):
        fg = cv2.resize(fg, (bg_w, bg_h))

    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg

    for c in range(0, 3):
        bg[:, :, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[:, :, c])
    return bg

# -------------------- GIF Frame Counter --------------------
frame_index = 0

# -------------------- Main Loop --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Default: awkward hamster frame
    right_overlay = awkward_hamster.copy()

    # Tongue detection
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            tongue_tip = face_landmarks.landmark[13]
            h, w, _ = frame.shape
            tx, ty = int(tongue_tip.x * w), int(tongue_tip.y * h)

            if ty > h * 0.7:
                # Use animated GIF frame
                overlay_rgba = tongue_frames[frame_index]
                frame_index = (frame_index + 1) % len(tongue_frames)
                # Convert RGBA to BGR for OpenCV
                right_overlay = cv2.cvtColor(overlay_rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    # Resize hamster/tongue to match webcam height
    right_overlay_resized = cv2.resize(right_overlay, (frame.shape[1], frame.shape[0]))

    # Split screen with 10px white padding
    padding = np.ones((frame.shape[0], 10, 3), dtype=np.uint8) * 255
    split_screen = np.hstack((frame, padding, right_overlay_resized))

    cv2.imshow("Split Screen Meme Camera", split_screen)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
