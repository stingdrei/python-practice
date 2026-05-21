import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

WINDOW_W      = 1280
WINDOW_H      = 720
BALL_RADIUS   = 16
PADDLE_W      = 20
PADDLE_H      = 120
BASE_SPEED    = 20.0
MAX_SPEED     = 60.0
SCORE_TO_WIN  = 7
HISTORY_LEN   = 6
CENTER_GAP    = 80
PADDLE_MIN_X  = PADDLE_W // 2 + 20
# Extra pixels added to EACH side of the paddle hit-box (invisible forgiveness zone)
COLLISION_PAD = 18
# Seconds the ball is immune to re-bounce after hitting a paddle
BOUNCE_COOLDOWN = 0.15

paddle_x = [80, WINDOW_W - 80]
paddle_y = [WINDOW_H // 2, WINDOW_H // 2]

ball_x    = float(WINDOW_W // 2)
ball_y    = float(WINDOW_H // 2)
ball_vx   = BASE_SPEED
ball_vy   = BASE_SPEED * 0.4 * random.choice([-1, 1])
prev_ball_x = ball_x
prev_ball_y = ball_y

last_bounce_time = [-999.0, -999.0]   # per-player cooldown timestamps

score       = [0, 0]
hand_history = {0: [], 1: []}
game_over   = False
winner      = None
score_flash = [0.0, 0.0]

_glow_buf    = None
_dark_frame  = None
_center_line = None

# Player names (filled in on the start screen)
player_names = ["Player 1", "Player 2"]


# ---------------------------------------------------------------------------
# Name-entry start screen — runs once before the game loop
# ---------------------------------------------------------------------------

def run_name_entry():
    """Blocking name-entry screen. Returns (name1, name2) or None to quit."""
    NAME_COLORS = [(0, 210, 255), (255, 90, 30)]   # cyan / orange
    BOX_W, BOX_H = 420, 60
    MAX_LEN = 18

    inputs      = ["", ""]
    active      = 0          # which field is focused
    blink_t     = time.time()

    # Fixed 1280×720 surface (no camera needed here)
    bg = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

    while True:
        canvas = bg.copy()

        # Title
        title = "HAND PING PONG"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.2, 4)
        cv2.putText(canvas, title, ((WINDOW_W - tw) // 2, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)

        sub = "Enter player names, then press ENTER to start"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(canvas, sub, ((WINDOW_W - sw) // 2, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)

        labels   = ["Player 1 (left side)", "Player 2 (right side)"]
        box_tops = [280, 430]

        blink_on = (time.time() - blink_t) % 1.0 < 0.5

        for i in range(2):
            bx = (WINDOW_W - BOX_W) // 2
            by = box_tops[i]
            color = NAME_COLORS[i]
            is_active = (i == active)

            # Label
            cv2.putText(canvas, labels[i], (bx, by - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

            # Box outline — thicker when active
            thickness = 2 if is_active else 1
            box_color = color if is_active else (80, 80, 80)
            cv2.rectangle(canvas, (bx, by), (bx + BOX_W, by + BOX_H),
                          box_color, thickness)

            # Text + blinking cursor
            display = inputs[i] + ("|" if is_active and blink_on else "")
            cv2.putText(canvas, display, (bx + 14, by + 42),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Ping Pong", canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord('q'):
            return None

        elif key == 9:                          # Tab — switch field
            active = 1 - active
            blink_t = time.time()

        elif key == 13:                         # Enter
            if active == 0 and inputs[0]:
                active  = 1
                blink_t = time.time()
            else:
                n0 = inputs[0].strip() or "Player 1"
                n1 = inputs[1].strip() or "Player 2"
                return n0, n1

        elif key == 8 or key == 127:            # Backspace
            if inputs[active]:
                inputs[active] = inputs[active][:-1]

        elif 32 <= key <= 126:                  # Printable ASCII
            if len(inputs[active]) < MAX_LEN:
                inputs[active] += chr(key)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def get_hand_center(lm, w, h):
    wrist   = lm.landmark[0]
    mid_mcp = lm.landmark[9]
    return (int((wrist.x + mid_mcp.x) / 2 * w),
            int((wrist.y + mid_mcp.y) / 2 * h))


def compute_velocity(history):
    if len(history) < 2:
        return 0.0, 0.0
    x1, y1, t1 = history[0]
    x2, y2, t2 = history[-1]
    dt = t2 - t1
    if dt < 1e-4:
        return 0.0, 0.0
    return (x2 - x1) / dt, (y2 - y1) / dt


def compute_acceleration(history):
    if len(history) < 3:
        return 0.0, 0.0
    mid = len(history) // 2
    vx1, vy1 = compute_velocity(history[:mid])
    vx2, vy2 = compute_velocity(history[mid:])
    t1 = history[mid][2] - history[0][2]
    t2 = history[-1][2] - history[mid][2]
    dt = (t1 + t2) / 2
    if dt < 1e-4:
        return 0.0, 0.0
    return (vx2 - vx1) / dt, (vy2 - vy1) / dt


def speed_from_hand(player_idx):
    hist = hand_history[player_idx]
    vx, vy = compute_velocity(hist)
    ax, ay = compute_acceleration(hist)
    hand_speed = math.hypot(vx, vy)
    hand_accel = math.hypot(ax, ay)
    accel_factor = min(hand_accel / 300.0, 1.0)
    speed_factor = min(hand_speed / 800.0, 0.6)
    return BASE_SPEED + max(accel_factor, speed_factor) * (MAX_SPEED - BASE_SPEED)


def reset_ball(direction=None):
    global ball_x, ball_y, ball_vx, ball_vy, prev_ball_x, prev_ball_y
    ball_x = float(WINDOW_W // 2)
    ball_y = float(random.randint(WINDOW_H // 4, 3 * WINDOW_H // 4))
    if direction is None:
        direction = random.choice([-1, 1])
    ball_vx = BASE_SPEED * direction
    ball_vy = BASE_SPEED * 0.4 * random.choice([-1, 1])
    prev_ball_x, prev_ball_y = ball_x, ball_y


# ---------------------------------------------------------------------------
# Collision — dual-check (CCD sweep + overlap fallback) with cooldown
# ---------------------------------------------------------------------------

def resolve_collision(player_idx, now):
    """
    Returns True and applies bounce if ball hits player's paddle.

    Strategy:
      Pass 1 — CCD sweep: did the ball path cross the paddle face this frame?
      Pass 2 — Overlap fallback: is the ball currently inside the expanded hitbox?
               (catches cases where the paddle moved into the ball)
    A per-player cooldown prevents double-bouncing.
    """
    global ball_x, ball_y, ball_vx, ball_vy

    if now - last_bounce_time[player_idx] < BOUNCE_COOLDOWN:
        return False

    px = paddle_x[player_idx]
    py = paddle_y[player_idx]

    # Expanded hit-box (invisible, more forgiving than the visual paddle)
    hit_hw = PADDLE_W  // 2 + COLLISION_PAD          # horizontal half-width
    hit_hh = PADDLE_H  // 2 + BALL_RADIUS + COLLISION_PAD  # vertical half-height

    if player_idx == 0:
        # Left paddle: ball hits right face, must be moving left
        face_x      = px + PADDLE_W // 2
        ball_edge   = ball_x - BALL_RADIUS
        prev_edge   = prev_ball_x - BALL_RADIUS
        moving_toward = ball_vx < 0

        # CCD: did left edge of ball cross the face line this frame?
        ccd_hit = moving_toward and (prev_edge >= face_x >= ball_edge)
        # Overlap: ball center currently inside the expanded hitbox
        overlap = (abs(ball_x - px) <= hit_hw and
                   abs(ball_y - py) <= hit_hh and
                   moving_toward)

    else:
        # Right paddle: ball hits left face, must be moving right
        face_x      = px - PADDLE_W // 2
        ball_edge   = ball_x + BALL_RADIUS
        prev_edge   = prev_ball_x + BALL_RADIUS
        moving_toward = ball_vx > 0

        ccd_hit = moving_toward and (prev_edge <= face_x <= ball_edge)
        overlap = (abs(ball_x - px) <= hit_hw and
                   abs(ball_y - py) <= hit_hh and
                   moving_toward)

    if not (ccd_hit or overlap):
        return False

    # Vertical range check — use the wider hit_hh so edge hits still register
    if abs(ball_y - py) > hit_hh:
        return False

    # --- Apply bounce ---
    new_speed = speed_from_hand(player_idx)
    offset    = float(np.clip((ball_y - py) / (PADDLE_H // 2), -1.0, 1.0))
    angle     = offset * math.radians(55)

    if player_idx == 0:
        ball_vx = new_speed * math.cos(angle)
        ball_vy = new_speed * math.sin(angle)
        # Eject ball fully outside the face so it can never be re-trapped
        ball_x  = float(face_x + BALL_RADIUS + 4)
    else:
        ball_vx = -(new_speed * math.cos(angle))
        ball_vy =   new_speed * math.sin(angle)
        ball_x  = float(face_x - BALL_RADIUS - 4)

    last_bounce_time[player_idx] = now
    score_flash[player_idx] = 0.3
    return True


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_glow_circle_fast(img, buf, center, radius, color):
    buf[:] = 0
    cv2.circle(buf, center, radius + 8, color, -1)
    cv2.GaussianBlur(buf, (15, 15), 5, buf)
    cv2.addWeighted(img, 1.0, buf, 0.5, 0, img)
    cv2.circle(img, center, radius, color, -1)


def draw_paddle_fast(img, buf, px, py, color, glow=False):
    x1 = px - PADDLE_W // 2
    y1 = py - PADDLE_H // 2
    x2 = px + PADDLE_W // 2
    y2 = py + PADDLE_H // 2
    if glow:
        buf[:] = 0
        cv2.rectangle(buf, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color, -1)
        cv2.GaussianBlur(buf, (15, 15), 6, buf)
        cv2.addWeighted(img, 1.0, buf, 0.55, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)


def build_center_line(h, w):
    cl = np.zeros((h, w, 3), dtype=np.uint8)
    for y_pos in range(0, h, 36):
        cv2.rectangle(cl, (w // 2 - 3, y_pos), (w // 2 + 3, y_pos + 20),
                      (180, 180, 180), -1)
    return cl


# ---------------------------------------------------------------------------
# Camera & main loop
# ---------------------------------------------------------------------------

# Show name-entry screen before opening the camera
_name_result = run_name_entry()
if _name_result is None:
    cv2.destroyAllWindows()
    raise SystemExit(0)
player_names[0], player_names[1] = _name_result

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)
cap.set(cv2.CAP_PROP_FPS,          60)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

prev_time   = time.time()
fps_display = 0.0
fps_counter = 0
fps_timer   = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    if _glow_buf is None:
        _glow_buf    = np.zeros((h, w, 3), dtype=np.uint8)
        _dark_frame  = np.zeros((h, w, 3), dtype=np.uint8)
        _center_line = build_center_line(h, w)

    left_x_min  = PADDLE_MIN_X
    left_x_max  = w // 2 - CENTER_GAP
    right_x_min = w // 2 + CENTER_GAP
    right_x_max = w - PADDLE_MIN_X

    # MediaPipe on half-res frame
    small  = cv2.resize(frame, (640, 360))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    cv2.addWeighted(frame, 0.18, _dark_frame, 0.82, 0, frame)
    canvas = frame

    now = time.time()
    dt  = now - prev_time
    prev_time = now

    fps_counter += 1
    fps_timer   += dt
    if fps_timer >= 0.5:
        fps_display = fps_counter / fps_timer
        fps_counter = fps_timer = 0

    left_hand_lm = right_hand_lm = None

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            cx = int((lm.landmark[0].x + lm.landmark[9].x) / 2 * w)
            cy = int((lm.landmark[0].y + lm.landmark[9].y) / 2 * h)
            if cx < w // 2:
                left_hand_lm  = (lm, cx, cy)
            else:
                right_hand_lm = (lm, cx, cy)

    if left_hand_lm:
        _, cx, cy    = left_hand_lm
        paddle_x[0]  = int(np.clip(cx, left_x_min,  left_x_max))
        paddle_y[0]  = int(np.clip(cy, PADDLE_H // 2, h - PADDLE_H // 2))
        hand_history[0].append((cx, cy, now))
        if len(hand_history[0]) > HISTORY_LEN:
            hand_history[0].pop(0)

    if right_hand_lm:
        _, cx, cy    = right_hand_lm
        paddle_x[1]  = int(np.clip(cx, right_x_min, right_x_max))
        paddle_y[1]  = int(np.clip(cy, PADDLE_H // 2, h - PADDLE_H // 2))
        hand_history[1].append((cx, cy, now))
        if len(hand_history[1]) > HISTORY_LEN:
            hand_history[1].pop(0)

    if not game_over:
        prev_ball_x, prev_ball_y = ball_x, ball_y

        ball_x += ball_vx
        ball_y += ball_vy

        # Top / bottom walls
        if ball_y - BALL_RADIUS <= 0:
            ball_y = float(BALL_RADIUS)
            ball_vy = abs(ball_vy)
        elif ball_y + BALL_RADIUS >= h:
            ball_y = float(h - BALL_RADIUS)
            ball_vy = -abs(ball_vy)

        # Paddle collisions (both checked every frame regardless of direction hint)
        hit0 = resolve_collision(0, now)
        hit1 = resolve_collision(1, now)

        # Score
        if ball_x < -BALL_RADIUS * 2:
            score[1] += 1
            score_flash[1] = 0.8
            if score[1] >= SCORE_TO_WIN:
                game_over = True
                winner    = 1
            else:
                reset_ball(direction=1)
        elif ball_x > w + BALL_RADIUS * 2:
            score[0] += 1
            score_flash[0] = 0.8
            if score[0] >= SCORE_TO_WIN:
                game_over = True
                winner    = 0
            else:
                reset_ball(direction=-1)

    score_flash[0] = max(0.0, score_flash[0] - dt)
    score_flash[1] = max(0.0, score_flash[1] - dt)

    # --- Render ---
    cv2.addWeighted(canvas, 1.0, _center_line, 0.6, 0, canvas)

    cv2.line(canvas, (left_x_max,  0), (left_x_max,  h), (0, 120, 200), 1)
    cv2.line(canvas, (right_x_min, 0), (right_x_min, h), (200, 60,  0), 1)

    draw_paddle_fast(canvas, _glow_buf, paddle_x[0], paddle_y[0],
                     (0, 210, 255), glow=score_flash[0] > 0)
    draw_paddle_fast(canvas, _glow_buf, paddle_x[1], paddle_y[1],
                     (255, 90, 30),  glow=score_flash[1] > 0)

    draw_glow_circle_fast(canvas, _glow_buf,
                          (int(ball_x), int(ball_y)), BALL_RADIUS, (255, 255, 255))

    # Speed HUD
    speed_p0 = speed_p1 = 0.0
    if len(hand_history[0]) >= 2:
        vx, vy = compute_velocity(hand_history[0])
        speed_p0 = math.hypot(vx, vy)
    if len(hand_history[1]) >= 2:
        vx, vy = compute_velocity(hand_history[1])
        speed_p1 = math.hypot(vx, vy)

    s0_color = (100, 255, 255) if score_flash[0] > 0 else (255, 255, 255)
    s1_color = (100, 180, 255) if score_flash[1] > 0 else (255, 255, 255)
    cv2.putText(canvas, str(score[0]), (w // 4 - 30, 85),
                cv2.FONT_HERSHEY_DUPLEX, 3.2, s0_color, 5, cv2.LINE_AA)
    cv2.putText(canvas, str(score[1]), (3 * w // 4 - 30, 85),
                cv2.FONT_HERSHEY_DUPLEX, 3.2, s1_color, 5, cv2.LINE_AA)

    cv2.putText(canvas, player_names[0], (20, h - 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"spd {speed_p0:.0f}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1, cv2.LINE_AA)
    n2_tw, _ = cv2.getTextSize(player_names[1], cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
    cv2.putText(canvas, player_names[1], (w - n2_tw - 20, h - 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 90, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"spd {speed_p1:.0f}", (w - 120, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FPS {fps_display:.0f}", (w // 2 - 30, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    if game_over:
        cv2.addWeighted(canvas, 0.5, _dark_frame, 0.5, 0, canvas)
        win_color = (0, 210, 255) if winner == 0 else (255, 90, 30)
        msg = f"{player_names[winner]} Wins!"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 2.8, 5)
        cv2.putText(canvas, msg, ((w - tw) // 2, h // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 2.8, win_color, 5, cv2.LINE_AA)
        sub = f"{score[0]}  -  {score[1]}"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(canvas, sub, ((w - sw) // 2, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
        hint = "Press R to play again   Q to quit"
        (hw_, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(canvas, hint, ((w - hw_) // 2, h // 2 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2, cv2.LINE_AA)

    cv2.imshow("Hand Ping Pong", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('r'):
        score        = [0, 0]
        game_over    = False
        winner       = None
        hand_history = {0: [], 1: []}
        last_bounce_time = [-999.0, -999.0]
        paddle_x[:]  = [80, w - 80]
        paddle_y[:]  = [h // 2, h // 2]
        reset_ball()

cap.release()
cv2.destroyAllWindows()
