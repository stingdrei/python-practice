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

Gestures:
    Left index only                        → monkey.jpeg
    Left 4 fingers + backface              → nub-nub-cat.gif
    Right index → lips + backface          → monkey2.jpeg
    Cross-arm dance x2 (L→R shoulder,
      then R→L shoulder, repeated twice)   → dog.mp4 (loops until
                                             dance stops, hides after 2s)
"""

import cv2
import mediapipe as mp
import time
import os
import numpy as np
from collections import deque
from typing import Optional
from PIL import Image

MONKEY_IMAGE_PATH  = "assets/monkey.jpeg"
MONKEY2_IMAGE_PATH = "assets/monkey2.jpeg"
NUBNUB_GIF_PATH    = "assets/nub-nub-cat.gif"
DOG_VIDEO_PATH     = "assets/dog.mp4"

# Video playback speed multiplier for dog.mp4.
#   1.0 = original speed
#   2.0 = twice as fast   (skip every other frame)
#   0.5 = half speed      (each frame shown twice as long)
#   Set to the native FPS of your video if you want exact control.
DOG_VIDEO_SPEED    = 5.0


# ══════════════════════════════════════════════
#  Asset loaders
# ══════════════════════════════════════════════

def load_static_image(path, display_size=(280, 280)):
    if not os.path.exists(path):
        print(f"[WARN] '{path}' not found.")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Could not read '{path}'.")
        return None
    return cv2.resize(img, display_size)


def load_gif_frames(path, display_size=(280, 280)):
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
#  Overlay helpers
# ══════════════════════════════════════════════

def overlay_image(frame, overlay, x, y):
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


def overlay_video_frame(frame, vid_frame, x, y, size=(320, 240)):
    """Resize and blit a plain BGR video frame onto the camera frame."""
    resized  = cv2.resize(vid_frame, size)
    fh, fw   = frame.shape[:2]
    vh, vw   = resized.shape[:2]
    x1, y1   = max(x, 0), max(y, 0)
    x2, y2   = min(x + vw, fw), min(y + vh, fh)
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return
    frame[y1:y2, x1:x2] = resized[sy1:sy2, sx1:sx2]


# ══════════════════════════════════════════════
#  Hand gesture detectors (unchanged)
# ══════════════════════════════════════════════

def is_palm_facing_away(hand_landmarks, handedness_label):
    lm      = hand_landmarks.landmark
    index_x = lm[5].x
    pinky_x = lm[17].x
    if handedness_label == "Right":
        return index_x > pinky_x
    else:
        return index_x < pinky_x


def is_left_index_only(extended, handedness):
    if handedness != "Left":
        return False
    return extended[1] and not any(extended[i] for i in [0, 2, 3, 4])


def is_left_four_no_thumb(extended, handedness, back_facing=False):
    if handedness != "Left":
        return False
    if not back_facing:
        return False
    return extended[0] and all(extended[i] for i in [1, 2, 3, 4])


def is_right_index_at_lips(hand_landmarks, handedness, face_landmarks, frame_w, frame_h):
    if handedness != "Right":
        return False
    if face_landmarks is None:
        return False
    tip = hand_landmarks.landmark[8]
    tx  = int(tip.x * frame_w)
    ty  = int(tip.y * frame_h)
    lip_ids = [13, 14, 61, 291, 78, 308, 402, 182]
    xs = [int(face_landmarks.landmark[i].x * frame_w) for i in lip_ids]
    ys = [int(face_landmarks.landmark[i].y * frame_h) for i in lip_ids]
    pad = 40
    return (min(xs) - pad <= tx <= max(xs) + pad and
            min(ys) - pad <= ty <= max(ys) + pad)


# ══════════════════════════════════════════════
#  Core finger counting
# ══════════════════════════════════════════════

def count_fingers(hand_landmarks, handedness_label):
    lm       = hand_landmarks.landmark
    tips     = [4,  8,  12, 16, 20]
    pips     = [3,  6,  10, 14, 18]
    extended = []
    if handedness_label == "Right":
        extended.append(lm[tips[0]].x < lm[pips[0]].x)
    else:
        extended.append(lm[tips[0]].x > lm[pips[0]].x)
    for i in range(1, 5):
        extended.append(lm[tips[i]].y < lm[pips[i]].y)
    return sum(extended), extended


# ══════════════════════════════════════════════
#  Cross-arm dance detector
# ══════════════════════════════════════════════
#
#  MediaPipe Pose landmark indices
#    11 = LEFT  shoulder   12 = RIGHT shoulder
#    15 = LEFT  wrist      16 = RIGHT wrist
#
#  IMPORTANT — coordinate space after cv2.flip(frame, 1):
#    The frame is horizontally flipped so it acts as a mirror.
#    MediaPipe Pose is called on the already-flipped image, so:
#      - x=0 is the LEFT edge of the *display* (person's RIGHT side)
#      - x=1 is the RIGHT edge of the *display* (person's LEFT side)
#    MediaPipe internally labels body parts relative to the person,
#    so lm[11] is truly the person's LEFT shoulder but its .x value
#    will be > 0.5 (right side of the display because of the flip).
#
#  Step A — left hand crosses to right shoulder:
#    Person moves their LEFT wrist (lm[15]) toward their RIGHT shoulder
#    (lm[12]).  After the flip, the right shoulder has a SMALLER x value
#    than the left shoulder, so crossing means:
#       lm[15].x  <  lm[12].x + CROSS_MARGIN
#
#  Step B — right hand crosses to left shoulder:
#    Person moves their RIGHT wrist (lm[16]) toward their LEFT shoulder
#    (lm[11]).  After the flip, the left shoulder has a LARGER x value,
#    so crossing means:
#       lm[16].x  >  lm[11].x - CROSS_MARGIN
#
#  Two complete A→B cycles within SEQUENCE_WINDOW seconds → trigger.

# How far past the opposite shoulder the wrist must travel (normalised coords).
# 0.0 = exactly at shoulder; positive = clearly crossed over.
CROSS_MARGIN     = 0.05
# How far above/below the shoulder the wrist is allowed to be (normalised).
SHOULDER_Y_TOL   = 0.30
# Seconds allowed for two full A→B cycles.
SEQUENCE_WINDOW  = 6.0
# Minimum time between registering the same step to avoid jitter.
DEBOUNCE_SEC     = 0.5
# Minimum MediaPipe visibility score for a landmark to be trusted.
VIS_THRESH       = 0.4


class DanceDetector:
    """Detects two consecutive L→R-shoulder / R→L-shoulder arm crosses."""

    STEP_A = "L_to_R"
    STEP_B = "R_to_L"

    def __init__(self):
        self._steps          = deque()
        self._last_step      = None
        self._last_step_time = 0.0

    # ── public ──────────────────────────────────

    def update(self, pose_lm):
        """
        Feed each frame's pose landmarks.
        Returns True the moment two A→B cycles are confirmed.
        Also returns the current raw diagnostic values as a dict.
        """
        diag = {
            "lw_x": None, "rw_x": None,
            "ls_x": None, "rs_x": None,
            "step_a": False, "step_b": False,
            "cycles": 0,
        }
        if pose_lm is None:
            return False, diag

        lm  = pose_lm.landmark
        now = time.time()

        lw = lm[15]   # left  wrist
        rw = lm[16]   # right wrist
        ls = lm[11]   # left  shoulder
        rs = lm[12]   # right shoulder

        diag["lw_x"] = round(lw.x, 3)
        diag["rw_x"] = round(rw.x, 3)
        diag["ls_x"] = round(ls.x, 3)
        diag["rs_x"] = round(rs.x, 3)

        # Visibility gate
        lw_ok = lw.visibility > VIS_THRESH
        rw_ok = rw.visibility > VIS_THRESH
        ls_ok = ls.visibility > VIS_THRESH
        rs_ok = rs.visibility > VIS_THRESH

        # Step A: left wrist crosses to right shoulder
        # (after flip: rs.x < ls.x, so wrist crossing means wrist.x < rs.x + margin)
        step_a = (lw_ok and rs_ok and
                  lw.x < rs.x + CROSS_MARGIN and
                  abs(lw.y - rs.y) < SHOULDER_Y_TOL)

        # Step B: right wrist crosses to left shoulder
        # (after flip: ls.x > rs.x, so wrist crossing means wrist.x > ls.x - margin)
        step_b = (rw_ok and ls_ok and
                  rw.x > ls.x - CROSS_MARGIN and
                  abs(rw.y - ls.y) < SHOULDER_Y_TOL)

        diag["step_a"] = step_a
        diag["step_b"] = step_b

        # Determine which step fired (prefer A over B if both for safety)
        step_now = None
        if step_a:
            step_now = self.STEP_A
        elif step_b:
            step_now = self.STEP_B

        if step_now:
            debounce_ok = (step_now != self._last_step or
                           now - self._last_step_time > DEBOUNCE_SEC)
            if debounce_ok:
                self._last_step      = step_now
                self._last_step_time = now
                self._steps.append((now, step_now))

        # Prune steps outside the window
        cutoff = now - SEQUENCE_WINDOW
        while self._steps and self._steps[0][0] < cutoff:
            self._steps.popleft()

        cycles = self._count_cycles()
        diag["cycles"] = cycles
        return cycles >= 2, diag

    def reset(self):
        self._steps.clear()
        self._last_step      = None
        self._last_step_time = 0.0

    # ── private ─────────────────────────────────

    def _count_cycles(self):
        """Count completed A→B pairs in chronological order."""
        cycles = 0
        want   = self.STEP_A
        for _, label in self._steps:
            if label == want:
                if want == self.STEP_A:
                    want = self.STEP_B
                else:
                    cycles += 1
                    want    = self.STEP_A
        return cycles


def is_currently_crossing(pose_lm):
    """
    True if EITHER arm is in a crossed position right now.
    Used to decide whether the dance is still ongoing.
    Simply checks if either wrist has passed the body's centre x.
    """
    if pose_lm is None:
        return False
    lm  = pose_lm.landmark
    lw  = lm[15]
    rw  = lm[16]
    ls  = lm[11]
    rs  = lm[12]
    mid = (ls.x + rs.x) / 2.0   # body midline in normalised x

    left_crossed  = (lw.visibility > VIS_THRESH and lw.x < mid)
    right_crossed = (rw.visibility > VIS_THRESH and rw.x > mid)
    return left_crossed or right_crossed


# ══════════════════════════════════════════════
#  MediaPipe setup
# ══════════════════════════════════════════════

mp_hands      = mp.solutions.hands
mp_face_mesh  = mp.solutions.face_mesh
mp_pose       = mp.solutions.pose
mp_draw       = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

COLOR_WHITE  = (255, 255, 255)
COLOR_GREEN  = (0,  220, 120)
COLOR_BLUE   = (0,   80, 220)
COLOR_YELLOW = (40, 210, 210)
COLOR_RED    = (60,  60, 220)


# ══════════════════════════════════════════════
#  Drawing helpers
# ══════════════════════════════════════════════

def draw_fps(frame, fps):
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

    # ── Load assets ───────────────────────────
    monkey_img    = load_static_image(MONKEY_IMAGE_PATH,  (280, 280))
    monkey2_img   = load_static_image(MONKEY2_IMAGE_PATH, (280, 280))
    nubnub_frames = load_gif_frames(NUBNUB_GIF_PATH, (280, 280))

    gif_frame_idx = 0
    gif_last_tick = time.time()
    GIF_FPS       = 15

    # ── Dog video ─────────────────────────────
    dog_cap        = None
    dog_active     = False
    dog_stop_time  = None
    DOG_LINGER_SEC = 2.0
    dog_last_tick  = 0.0
    dog_cur_frame  = None
    dog_native_fps = 30.0     # overwritten after file opens

    if os.path.exists(DOG_VIDEO_PATH):
        dog_cap = cv2.VideoCapture(DOG_VIDEO_PATH)
        if not dog_cap.isOpened():
            print(f"[WARN] Could not open '{DOG_VIDEO_PATH}'.")
            dog_cap = None
        else:
            dog_native_fps = dog_cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"[INFO] Loaded '{DOG_VIDEO_PATH}' — native FPS: {dog_native_fps:.1f}.")
    else:
        print(f"[WARN] '{DOG_VIDEO_PATH}' not found — dog video disabled.")

    dance_detector  = DanceDetector()
    dance_triggered = False   # True once two cycles detected, reset when dance ends
    last_diag       = {"lw_x": None, "rw_x": None,
                       "ls_x": None, "rs_x": None,
                       "step_a": False, "step_b": False, "cycles": 0}
    
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
    ) as face_mesh, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        print("=" * 60)
        print("  Finger Counter   |  Q/ESC → quit   S → screenshot")
        print("  Cross-arm dance: L-hand→R-shoulder, R-hand→L-shoulder")
        print("  Repeat twice within 6 s to trigger dog.mp4")
        print("=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Run all detectors ─────────────────────────────
            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            face_lm = (face_results.multi_face_landmarks[0]
                       if face_results.multi_face_landmarks else None)
            pose_lm = (pose_results.pose_landmarks
                       if pose_results.pose_landmarks else None)

            # ── Dance state machine ───────────────────────────
            if dog_cap is not None:
                triggered, last_diag = dance_detector.update(pose_lm)

                if triggered and not dance_triggered:
                    dance_triggered = True
                    dog_active      = True
                    dog_stop_time   = None
                    dog_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("[INFO] Dance confirmed! Playing dog.mp4 …")

                if dance_triggered and dog_active:
                    still_going = is_currently_crossing(pose_lm)

                    if not still_going:
                        if dog_stop_time is None:
                            dog_stop_time = now
                            print("[INFO] Dance paused — 2 s linger started.")
                        elif now - dog_stop_time >= DOG_LINGER_SEC:
                            dog_active      = False
                            dance_triggered = False
                            dog_stop_time   = None
                            dance_detector.reset()
                            dog_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            print("[INFO] Dance ended. dog.mp4 hidden.")
                    else:
                        if dog_stop_time is not None:
                            dog_stop_time = None   # dance resumed
            else:
                # No video file — still update diag for HUD
                _, last_diag = dance_detector.update(pose_lm)

            # ── Hand gesture flags ────────────────────────────
            total_fingers = 0
            show_monkey   = False
            show_monkey2  = False
            show_nubnub   = False

            if hand_results.multi_hand_landmarks:
                for hand_lm, hand_info in zip(
                        hand_results.multi_hand_landmarks,
                        hand_results.multi_handedness):
                    handedness  = hand_info.classification[0].label
                    back_facing = is_palm_facing_away(hand_lm, handedness)
                    cnt, extended = count_fingers(hand_lm, handedness)

                    if not back_facing:
                        total_fingers += cnt

                    if back_facing:
                        if is_right_index_at_lips(hand_lm, handedness,
                                                  face_lm, w, h):
                            show_monkey2 = True
                    else:
                        if is_left_index_only(extended, handedness):
                            show_monkey = True

                    if is_left_four_no_thumb(extended, handedness, back_facing):
                        show_nubnub = True

            # ── FPS ───────────────────────────────────────────
            draw_fps(frame, fps)

            # ── Monkey overlay ────────────────────────────────
            if show_monkey and monkey_img is not None:
                mh, mw = monkey_img.shape[:2]
                overlay_image(frame, monkey_img,
                              w - mw - 20, (h - mh) // 2)

            # ── Monkey2 overlay ───────────────────────────────
            if show_monkey2 and monkey2_img is not None:
                mh, mw = monkey2_img.shape[:2]
                overlay_image(frame, monkey2_img,
                              w - mw - 20, (h - mh) // 2)

            # ── Nub-Nub Cat GIF overlay ───────────────────────
            if show_nubnub and nubnub_frames:
                if now - gif_last_tick > 1.0 / GIF_FPS:
                    gif_frame_idx = (gif_frame_idx + 1) % len(nubnub_frames)
                    gif_last_tick = now
                gframe = nubnub_frames[gif_frame_idx]
                gh, gw = gframe.shape[:2]
                overlay_image(frame, gframe, w - gw - 4, (h - gh) // 2)

            # ── Dog video overlay ─────────────────────────────
            if dog_active and dog_cap is not None:
                interval = 1.0 / (dog_native_fps * DOG_VIDEO_SPEED)

                if now - dog_last_tick >= interval:
                    ret_v, vid_frame = dog_cap.read()
                    if not ret_v:                   # end of video → loop
                        dog_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_v, vid_frame = dog_cap.read()
                    if ret_v:
                        dog_cur_frame = vid_frame   # cache latest frame
                    dog_last_tick = now

                if dog_cur_frame is not None:
                    VID_W, VID_H = 360, 270
                    vx = w - VID_W - 20
                    vy = (h - VID_H) // 2
                    overlay_video_frame(frame, dog_cur_frame, vx, vy, (VID_W, VID_H))



            cv2.imshow("Finger Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("[INFO] Exiting.")
                break
            elif key == ord('s'):
                os.makedirs(screenshot_dir, exist_ok=True)
                fname = os.path.join(
                    screenshot_dir, f"screenshot_{int(time.time())}.png")
                cv2.imwrite(fname, frame)
                print(f"[INFO] Screenshot saved → {fname}")

    cap.release()
    if dog_cap is not None:
        dog_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()