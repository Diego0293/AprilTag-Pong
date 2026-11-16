import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import math
import random
import time
import tkinter as tk

import cv2
import numpy as np
import pygame
from pupil_apriltags import Detector

# ----------------------------------------------------------------- settings
LEFT_TAG_ID = 0
RIGHT_TAG_ID = 1
GAME_W = 1280
SHOW_TAG_OUTLINE = True

num_cores = os.cpu_count() or 1
cv2.setNumThreads(num_cores)
print("Using", num_cores, "cores")

detector = Detector(families="tag36h11", nthreads=num_cores)


# --------------------------------------------------------------------- game
class Pong:
    def __init__(self, w, h):
        pygame.font.init()
        self.w, self.h = w, h
        self.surface = pygame.Surface((w, h))
        self.font = pygame.font.Font(None, int(h * 0.14))
        self.pad_w = max(8, int(w * 0.012))
        self.pad_h = int(h * 0.22)
        self.margin = int(w * 0.03)
        self.ball_r = max(6, int(h * 0.02))
        self.ball_speed = w * 0.4
        self.score = [0, 0]
        self.left_y = self.right_y = h / 2
        self.reset_ball(random.choice((-1, 1)))

    def reset_ball(self, direction):
        self.bx, self.by = self.w / 2, self.h / 2
        ang = random.uniform(-0.5, 0.5)
        self.vx = direction * self.ball_speed * math.cos(ang)
        self.vy = self.ball_speed * math.sin(ang)

    def reset_score(self):
        self.score = [0, 0]
        self.reset_ball(random.choice((-1, 1)))

    def _clamp(self, y):
        half = self.pad_h / 2
        return max(half, min(self.h - half, y))

    def set_paddle(self, side, y):
        setattr(self, side + "_y", self._clamp(y))

    def update(self, dt):
        self.bx += self.vx * dt
        self.by += self.vy * dt
        r = self.ball_r

        if self.by - r < 0:
            self.by, self.vy = r, abs(self.vy)
        elif self.by + r > self.h:
            self.by, self.vy = self.h - r, -abs(self.vy)

        lx = self.margin + self.pad_w
        rx = self.w - self.margin - self.pad_w
        half = self.pad_h / 2 + r
        vmax = self.ball_speed * 1.8
        if self.vx < 0 and self.bx - r <= lx and abs(self.by - self.left_y) <= half:
            self.bx = lx + r
            self.vx = abs(self.vx) * 1.05
            self.vy += (self.by - self.left_y) * 4
        elif self.vx > 0 and self.bx + r >= rx and abs(self.by - self.right_y) <= half:
            self.bx = rx - r
            self.vx = -abs(self.vx) * 1.05
            self.vy += (self.by - self.right_y) * 4
        self.vx = max(-vmax, min(vmax, self.vx))
        self.vy = max(-vmax, min(vmax, self.vy))

        if self.bx < 0:
            self.score[1] += 1
            self.reset_ball(1)
        elif self.bx > self.w:
            self.score[0] += 1
            self.reset_ball(-1)

    def draw(self):
        """Draw on black; black = transparent when composited."""
        s = self.surface
        s.fill((0, 0, 0))
        for y in range(0, self.h, int(self.h * 0.06)):
            pygame.draw.rect(s, (200, 200, 220),
                             (self.w // 2 - 3, y, 6, int(self.h * 0.03)))
        pygame.draw.rect(s, (80, 220, 255),
                         (self.margin, self.left_y - self.pad_h / 2, self.pad_w, self.pad_h))
        pygame.draw.rect(s, (255, 120, 80),
                         (self.w - self.margin - self.pad_w, self.right_y - self.pad_h / 2,
                          self.pad_w, self.pad_h))
        pygame.draw.circle(s, (255, 255, 255), (int(self.bx), int(self.by)), self.ball_r)
        for i, x in enumerate((self.w * 0.25, self.w * 0.75)):
            img = self.font.render(str(self.score[i]), True, (255, 255, 255))
            s.blit(img, img.get_rect(center=(x, self.h * 0.1)))
        return s


def pygame_surface_to_bgr(surf):
    arr = pygame.surfarray.array3d(surf)          # (w, h, 3) RGB
    arr = np.ascontiguousarray(arr.transpose(1, 0, 2))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def overlay_game(frame, game_bgr):
    """Composite non-black game pixels over the frame (in place)."""
    fh, fw = frame.shape[:2]
    big = cv2.resize(game_bgr, (fw, fh), interpolation=cv2.INTER_LINEAR)
    mask = big.max(axis=2) > 20
    frame[mask] = big[mask]


def pick_paddle_tags(detections):
    """Return (left_tag, right_tag); either may be None."""
    if LEFT_TAG_ID is not None and RIGHT_TAG_ID is not None:
        left = next((t for t in detections if t.tag_id == LEFT_TAG_ID), None)
        right = next((t for t in detections if t.tag_id == RIGHT_TAG_ID), None)
        return left, right
    if not detections:
        return None, None
    by_x = sorted(detections, key=lambda t: t.center[0])
    if len(by_x) == 1:
        t = by_x[0]
        return (t, None) if t.center[0] < cam_w / 2 else (None, t)
    return by_x[0], by_x[-1]


# ------------------------------------------------------------------- camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
camera.set(cv2.CAP_PROP_FPS, 30)

cam_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera resolution:", cam_w, "x", cam_h)

text_scale = cam_w / 1280
text_thick = max(1, int(2 * text_scale))
line_thick = max(1, int(3 * text_scale))

root = tk.Tk()
screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
root.destroy()
scale = min(screen_w * 0.9 / cam_w, screen_h * 0.9 / cam_h)
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", int(cam_w * scale), int(cam_h * scale))

# -------------------------------------------------------------------- main
game_h = int(GAME_W * cam_h / cam_w)
game = Pong(GAME_W, game_h)
last_t = time.perf_counter()

while True:
    key = cv2.waitKey(1) & 0xFF
    ok, frame = camera.read()
    if not ok:
        print("Cam not Working :(")
        break
    if key == ord("q") or cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord(" "):
        game.reset_score()

    now = time.perf_counter()
    dt = min(now - last_t, 0.05)
    last_t = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    left_tag, right_tag = pick_paddle_tags(detections)

    for tag in detections:
        if SHOW_TAG_OUTLINE:
            pts = tag.corners.astype(int)
            col = (0, 0, 255) if tag in (left_tag, right_tag) else (120, 120, 120)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), col, line_thick)

    # paddle follows its tag's vertical position (camera px -> game px)
    for side, tag in (("left", left_tag), ("right", right_tag)):
        if tag is not None:
            game.set_paddle(side, tag.center[1] * game_h / cam_h)

    both_visible = left_tag is not None and right_tag is not None
    if both_visible:
        game.update(dt)
    overlay_game(frame, pygame_surface_to_bgr(game.draw()))

    status = (f"left tag {'OK' if left_tag else 'MISSING'}   "
              f"right tag {'OK' if right_tag else 'MISSING'}   [space] reset  [q] quit")
    if not both_visible:
        cv2.putText(frame, "Show both paddle tags to play",
                    (int(40 * text_scale), int(60 * text_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * text_scale, (0, 0, 255), text_thick)
    cv2.putText(frame, status, (int(40 * text_scale), cam_h - int(30 * text_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * text_scale, (255, 255, 255), text_thick)

    cv2.imshow("Webcam", frame)

camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
pygame.quit()
