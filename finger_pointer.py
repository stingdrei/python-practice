"""
Finger Counter using MediaPipe Hand Landmark Detection
=======================================================
Requirements:
    pip install opencv-python mediapipe pillow

Run:
    python finger_counter.py

Controls:
    Q or ESC  - Quit
    S         - Save screenshot
"""

import cv2
import mediapipe as mp
import time
import os
import numpy as np
from PIL import Image

MONKEY_IMAGE_PATH  = "monkey.jpeg"
MONKEY2_IMAGE_PATH = "monkey2.jpeg"
NUBNUB_GIF_PATH    = "nub-nub-cat.gif"


# ══════════════════════════════════════════════
#  Asset loaders
# ══════════════════════════════════════════════

def load_static_image(path: str, display_size=(280, 280)):
    if not os.path.exists(path):
        print(f"[WARN] '{path}' not found.")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Could not read '{path}'.")
        return None
    return cv2.resize(img, display_size)


def load_gif_frames(path: str, display_size=(280, 280)) -> list:
    if not os.path.exists(path):
        print(f"[WARN] '{path}' not found.")
        return []
    try:
        gif    = Image.open(path)
        frames = []
        try:
            while True:
                frame_rgba = gif.convert("RGBA")
                arr  = np.array(frame_rgba)
                bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                bgra = cv2.resize(bgra, display_size)
                frames.append(bgra)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        print(f"[INFO] Loaded '{path}' — {len(frames)} frames.")
        return frames
    except Exception as e:
        print(f"[WARN] Could not load GIF '{path}': {e}")
        return []


# ══════════════════════════════════════════════
#  Overlay helper
# ══════════════════════════════════════════════

def overlay_image(frame, overlay, x: int, y: int):
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, fw), min(y + oh, fh)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    src = overlay[oy1:oy2, ox1:ox2]
    if src.shape[2] == 4:
        alpha   = src[:, :, 3:4].astype(np.float32) / 255.0
        rgb     = src[:, :, :3].astype(np.float32)
        blended = (alpha * rgb + (1.0 - alpha) * roi.astype(np.float32)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
    else:
        frame[y1:y2, x1:x2] = src


# ══════════════════════════════════════════════
#  Gesture detectors
# ══════════════════════════════════════════════

def is_palm_facing_away(hand_landmarks, handedness_label: str) -> bool:
    lm = hand_landmarks.landmark
    index_x = lm[5].x
    pinky_x  = lm[17].x
    if handedness_label == "Right":
        return index_x > pinky_x
    else:
        return index_x < pinky_x


def is_left_index_only(extended: list, handedness: str) -> bool:
    if handedness != "Left":
        return False
    return extended[1] and not any(extended[i] for i in [0, 2, 3, 4])


def is_left_four_no_thumb(extended: list, handedness: str, back_facing: bool = False) -> bool:
    if handedness != "Left":
        return False
    if not back_facing:
        return False
    return extended[0] and all(extended[i] for i in [1, 2, 3, 4])


def is_right_index_at_lips(hand_landmarks, handedness: str,
                            face_landmarks, frame_w: int, frame_h: int) -> bool:
    """
    Returns True when the RIGHT hand index fingertip (lm[8]) is close
    to the mouth/lips region of the detected face.

    Lips region is estimated from face mesh landmarks:
      - Upper lip top center : lm[13]
      - Lower lip bottom center: lm[14]
      - Left mouth corner : lm[61]
      - Right mouth corner: lm[291]
    We expand the bounding box by a padding factor so the trigger
    isn't pixel-perfect.
    """
    if handedness != "Right":
        return False
    if face_landmarks is None:
        return False

    # Index fingertip in pixel coords
    tip = hand_landmarks.landmark[8]
    tx  = int(tip.x * frame_w)
    ty  = int(tip.y * frame_h)

    # Lips bounding box from face mesh
    lip_ids = [13, 14, 61, 291, 78, 308, 402, 182]
    xs = [int(face_landmarks.landmark[i].x * frame_w) for i in lip_ids]
    ys = [int(face_landmarks.landmark[i].y * frame_h) for i in lip_ids]

    pad    = 40   # pixel padding around lips region
    lip_x1 = min(xs) - pad
    lip_x2 = max(xs) + pad
    lip_y1 = min(ys) - pad
    lip_y2 = max(ys) + pad

    return lip_x1 <= tx <= lip_x2 and lip_y1 <= ty <= lip_y2


# ══════════════════════════════════════════════
#  MediaPipe setup
# ══════════════════════════════════════════════
mp_hands      = mp.solutions.hands
mp_face_mesh  = mp.solutions.face_mesh
mp_draw       = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

FINGER_TIPS  = [4,  8,  12, 16, 20]
FINGER_PIPS  = [3,  6,  10, 14, 18]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

COLOR_WHITE  = (255, 255, 255)
COLOR_COUNT  = (0,  220, 120)
COLOR_ZERO   = (0,   80, 220)


# ══════════════════════════════════════════════
#  Core finger counting
# ══════════════════════════════════════════════

def count_fingers(hand_landmarks, handedness_label: str):
    lm = hand_landmarks.landmark
    extended = []
    if handedness_label == "Right":
        extended.append(lm[FINGER_TIPS[0]].x < lm[FINGER_PIPS[0]].x)
    else:
        extended.append(lm[FINGER_TIPS[0]].x > lm[FINGER_PIPS[0]].x)
    for i in range(1, 5):
        extended.append(lm[FINGER_TIPS[i]].y < lm[FINGER_PIPS[i]].y)
    return sum(extended), extended


# ══════════════════════════════════════════════
#  Drawing helpers
# ══════════════════════════════════════════════

def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                COLOR_WHITE, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    prev_time      = time.time()
    screenshot_dir = "screenshots"

    # Load assets once
    monkey_img      = load_static_image(MONKEY_IMAGE_PATH,  (280, 280))
    monkey2_img     = load_static_image(MONKEY2_IMAGE_PATH, (280, 280))
    nubnub_frames   = load_gif_frames(NUBNUB_GIF_PATH, (280, 280))

    gif_frame_idx   = 0
    gif_last_tick   = time.time()
    GIF_FPS         = 15

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        print("=" * 55)
        print("  Finger Counter   |  Q/ESC → quit   S → screenshot")
        print("  Gestures:")
        print("    Left index only            → monkey.jpeg")
        print("    Left 4 fingers + backface  → nub-nub-cat.gif")
        print("    Right index → lips         → monkey2.jpeg")
        print("=" * 55)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands and face mesh
            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Get first detected face (if any)
            face_lm = None
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0]

            total_fingers = 0
            num_hands     = 0
            show_monkey   = False
            show_monkey2  = False
            show_nubnub   = False

            if hand_results.multi_hand_landmarks:
                num_hands = len(hand_results.multi_hand_landmarks)

                for idx, (hand_lm, hand_info) in enumerate(
                    zip(hand_results.multi_hand_landmarks,
                        hand_results.multi_handedness)
                ):
                    handedness  = hand_info.classification[0].label
                    back_facing = is_palm_facing_away(hand_lm, handedness)
                    count, extended = count_fingers(hand_lm, handedness)

                    if not back_facing:
                        total_fingers += count

                    if back_facing:
                        pass
                    else:
                        if is_left_index_only(extended, handedness):
                            show_monkey = True
                        if is_right_index_at_lips(hand_lm, handedness, face_lm, w, h):
                            show_monkey2 = True

                    if is_left_four_no_thumb(extended, handedness, back_facing):
                        show_nubnub = True

                    # Skeleton
                    mp_draw.draw_landmarks(
                        frame, hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw_style.get_default_hand_landmarks_style(),
                        mp_draw_style.get_default_hand_connections_style(),
                    )

            # ── FPS ───────────────────────────────────────────
            draw_fps(frame, fps)

            # ── Monkey overlay (right side) ───────────────────
            if show_monkey:
                if monkey_img is not None:
                    mh, mw = monkey_img.shape[:2]
                    mx = w - mw - 20
                    my = (h - mh) // 2
                    overlay_image(frame, monkey_img, mx, my)
                else:
                    cv2.putText(frame, "monkey.jpeg not found!",
                                (w // 2 - 160, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 100, 255), 2, cv2.LINE_AA)

            # ── Monkey2 overlay (left side) — right index → lips
            if show_monkey2:
                if monkey2_img is not None:
                    mh, mw = monkey2_img.shape[:2]
                    mx = 20
                    my = (h - mh) // 2
                    overlay_image(frame, monkey2_img, mx, my)
                else:
                    cv2.putText(frame, "monkey2.jpeg not found!",
                                (w // 2 - 160, h // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 100, 255), 2, cv2.LINE_AA)

            # ── Nub-Nub Cat GIF overlay (center, triggered) ──
            if show_nubnub:
                if nubnub_frames:
                    if now - gif_last_tick > 1.0 / GIF_FPS:
                        gif_frame_idx = (gif_frame_idx + 1) % len(nubnub_frames)
                        gif_last_tick = now
                    gframe = nubnub_frames[gif_frame_idx]
                    gh, gw = gframe.shape[:2]
                    gx = (w - gw) // 2
                    gy = (h - gh) // 2
                    overlay_image(frame, gframe, gx, gy)
                else:
                    cv2.putText(frame, "nub-nub-cat.gif not found!",
                                (w // 2 - 200, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 100, 255), 2, cv2.LINE_AA)

            cv2.imshow("Finger Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("[INFO] Exiting.")
                break
            elif key == ord('s'):
                os.makedirs(screenshot_dir, exist_ok=True)
                fname = os.path.join(screenshot_dir,
                                     f"screenshot_{int(time.time())}.png")
                cv2.imwrite(fname, frame)
                print(f"[INFO] Screenshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()