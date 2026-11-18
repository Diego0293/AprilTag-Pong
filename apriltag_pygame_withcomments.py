import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import math
import threading
import time
import tkinter as tk

import cv2
import numpy as np
import pygame
from pupil_apriltags import Detector

# --------------------------------------------------------------- settings
# BETTER PERFORMANCE WHEN PLUGGED IN, MODIFIED TO RUN BETTER ON POWER
# Camera settings. 1080p is fine, 4k causes lag
CAPTURE_W = 1920
CAPTURE_H = 1080
CAPTURE_FPS = 30

# Width of the game, the detection picture and the window, in pixels.
# Bigger = sharper picture and tags found from farther away, but slower.
GAME_W = 1280
# Frames per second of the game
GAME_FPS = 60

# 1.0 = the platform is exactly the size of the tag. Bigger makes it easier to hit the ball.
HIT_SCALE = 1.0
# How long (seconds) a platform stays after its tag disappears from view.
PLATFORM_HOLD = 0.25
# How long (seconds) a platform takes to catch up to its tag. 0 = jumps straight to it, higher = slower.
PLATFORM_FOLLOW_TIME = 0.03
# Smooths how fast the platform seems to move. 1 = no smoothing, lower = smoother.
PLATFORM_VEL_SMOOTH = 0.5
# True = draw outlines around the ball and platforms so you can see the hit areas.
DEBUG = True

BASE_GRAVITY = 1000                 # how fast the ball speeds up downward
PLATFORM_INFLUENCE = 0.3            # how much a moving platform adds to the ball's speed
BASE_BALL_SPEED = 400               # the slowest the ball is allowed to go
BASE_MAX_BALL_SPEED = 1500          # the fastest the ball is allowed to go
AIR_RESISTANCE = 0.3                # slows the ball down a little over time
BASE_MAX_PLATFORM_SPEED = 4000      # ignore platform speeds above this (detection glitches) 
MAX_PLATFORM_ROT_SPEED = 2000       # same idea for spinning, in degrees per second

# Each game frame is cut into this many tiny steps so fast things can't skip through each other.
SUBSTEPS = 10
# When the ball overlaps a platform, push it out by at most this many pixels.
BASE_MAX_PUSH_OUT = 60

# The dotted line that shows where the ball is going.
PREDICTION_TIME = 1.0               # how many seconds ahead it looks
PREDICTION_STEPS = 120              # how many dots it draws
TRAJECTORY_ALPHA = 110              # how see-through the dots are (0-255)

# Colors, written as (red, green, blue).
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PLATFORM_COLOR = (80, 220, 255)
PLATFORM_ALPHA = 140                # how see-through the platform fill is (0-255)

# Folder next to this script where jumpy.png can be put.
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ------------------------------------------------------------ tag detector
# Use every CPU core for finding tags.
num_cores = os.cpu_count() or 1
cv2.setNumThreads(num_cores)
print("Using", num_cores, "cores")
# "tag36h11" is the AprilTag family we are using.
detector = Detector(families="tag36h11", nthreads=num_cores)

# ---------------------------------------------------------------- camera
# Open the first camera and ask for the settings above.
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
camera.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
# Keep only the newest frame so the picture doesn't fall behind, helps with lag.
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cam_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera resolution:", cam_w, "x", cam_h)

# ------------------------------------------------------- game size and scale
SCREEN_WIDTH = GAME_W
SCREEN_HEIGHT = int(GAME_W * cam_h / cam_w)
SCALE = SCREEN_HEIGHT / 500.0

# Multiply the physics numbers by SCALE so the game feels the same if windowed or fullscreen
GRAVITY = BASE_GRAVITY * SCALE
BALL_SPEED = BASE_BALL_SPEED * SCALE
MAX_BALL_SPEED = BASE_MAX_BALL_SPEED * SCALE
MAX_PLATFORM_SPEED = BASE_MAX_PLATFORM_SPEED * SCALE
MAX_PUSH_OUT = int(BASE_MAX_PUSH_OUT * SCALE)
BALL_SIZE = int(40 * SCALE)
TRAJECTORY_DOT = max(2, int(2 * SCALE))

# ---------------------------------------------------------------- pygame
pygame.display.init()
# pygame needs a window to exist before it can load images. This one is 1 pixel and never shown.
pygame.display.set_mode((1, 1))

# The game is drawn on this see-through layer.
screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)


def make_fallback_ball():
    # Draws a yellow ball with a dark outline. Used when jumpy.png isnt found, used for testing.
    surf = pygame.Surface((BALL_SIZE, BALL_SIZE), pygame.SRCALPHA)
    c = BALL_SIZE // 2
    pygame.draw.circle(surf, (255, 200, 0), (c, c), c)
    pygame.draw.circle(surf, (40, 40, 40), (c, c), c, max(2, BALL_SIZE // 12))
    return surf


def load_ball_image():
    # Try to load assets/jumpy.png and resize it. If that fails, draw a plain ball instead.
    path = os.path.join(ASSET_DIR, "jumpy.png")
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(img, (BALL_SIZE, BALL_SIZE))
    except (FileNotFoundError, pygame.error):
        print("assets/jumpy.png not found, drawing a plain ball instead")
        return make_fallback_ball()


jumpy_image = load_ball_image()


# ------------------------------------------------- collision and physics
def draw_mask_outline(mask, rect, color):
    # Draws a line around the solid pixels of a mask so you can see the hit area.
    points = mask.outline()
    screen_points = []

    for point in points:
        # Move each outline point from "inside the mask" to "where it is on screen".
        screen_points.append(
            (point[0] + rect.x, point[1] + rect.y)
        )

    if len(screen_points) > 1:
        pygame.draw.lines(screen, color, True, screen_points, 2)


def check_platform_collision(ball_rect, ball_mask, platform):
    # only builds the platform's mask when the ball is close, helps with performance.
    if not ball_rect.colliderect(platform.rect):
        return False
    offset = (
        ball_rect.x - platform.rect.x,
        ball_rect.y - platform.rect.y
    )

    # Checks if solid pixels of the ball and the platform overlap
    return platform.mask.overlap(ball_mask, offset) is not None


def apply_air_resistance(vel_x, vel_y, delta_time):
    # Slows the ball slightly, but keeps its speed between the minimum and maximum.

    speed = math.sqrt(vel_x ** 2 + vel_y ** 2)

    # If the ball is stopped, gives it a starting push up and to the right.
    if speed == 0:
        return BALL_SPEED / math.sqrt(2), -BALL_SPEED / math.sqrt(2)

    # Lose a little speed.
    drag_factor = max(0, 1 - AIR_RESISTANCE * delta_time)
    new_speed = speed * drag_factor

    # Never go slower than BALL_SPEED or faster than MAX_BALL_SPEED.
    new_speed = max(BALL_SPEED, min(new_speed, MAX_BALL_SPEED))

    # Change the speed but keep the ball heading the same way.
    scale = new_speed / speed

    vel_x *= scale
    vel_y *= scale

    return vel_x, vel_y


def calculate_platform_velocity(platform, ball_rect):
    # checks how far the ball is from the platform's center.
    rx = ball_rect.centerx - platform.center[0]
    ry = ball_rect.centery - platform.center[1]

    # Spin speed, converted from degrees to radians per second.
    omega = math.radians(platform.angular_velocity)

    # Extra speed at the ball's position caused by the spinning.
    rotational_vel_x = omega * ry
    rotational_vel_y = -omega * rx

    # Total = the platform moving as a whole + the spinning part.
    platform_vel_x = platform.vel_x + rotational_vel_x
    platform_vel_y = platform.vel_y + rotational_vel_y

    return platform_vel_x, platform_vel_y


def get_platform_normal(ball_rect, platform):
    # Works out which direction to push the ball so it leaves the tag.
    # It finds the closest edge of the tag to the ball and points away from it.

    px, py = ball_rect.center
    cx, cy = platform.center
    quad = platform.quad

    best = None                         # the closest edge found so far
    inside = True                       # is the ball center inside the tag?

    # Look at each of the four edges.
    for i in range(4):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % 4]
        abx, aby = bx - ax, by - ay
        length_sq = abx * abx + aby * aby

        if length_sq < 1e-9:
            continue

        length = math.sqrt(length_sq)

        nx, ny = aby / length, -abx / length
        mx, my = (ax + bx) / 2, (ay + by) / 2
        if nx * (mx - cx) + ny * (my - cy) < 0:
            nx, ny = -nx, -ny

        # If the ball is on the outer side of any edge, it is not inside the tag.
        if (px - ax) * nx + (py - ay) * ny > 0:
            inside = False

        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
        qx, qy = ax + t * abx, ay + t * aby
        dist = math.hypot(px - qx, py - qy)

        if best is None or dist < best[0]:
            best = (dist, qx, qy, nx, ny)

    # No usable edge: just push straight up.
    if best is None:
        return 0.0, -1.0

    dist, qx, qy, nx, ny = best

    # Ball center is inside the tag (or exactly on the edge): push out along the edge's direction.
    if inside or dist < 1e-6:
        return float(nx), float(ny)

    # Ball is outside: push away from the closest point on the tag.
    return float((px - qx) / dist), float((py - qy) / dist)


def calculate_bounce(vel_x, vel_y, normal_x, normal_y,
                     platform_vel_x, platform_vel_y):
    # Works out the ball's new speed and direction after it hits a platform.
    relative_vel_x = vel_x - platform_vel_x
    relative_vel_y = vel_y - platform_vel_y

    # Keep that same speed after the bounce.
    speed = math.sqrt(relative_vel_x ** 2 + relative_vel_y ** 2)

    # The ball leaves straight away from the surface at that speed, and adds some speed from platform.
    vel_x = normal_x * speed + platform_vel_x * PLATFORM_INFLUENCE
    vel_y = normal_y * speed + platform_vel_y * PLATFORM_INFLUENCE

    return vel_x, vel_y


def handle_platform_collision(x, y, vel_x, vel_y,
                              ball_rect, ball_mask, platform):
    # Checks one platform. If the ball is touching it, push the ball out and bounce it.

    if not check_platform_collision(ball_rect, ball_mask, platform):
        return x, y, vel_x, vel_y

    normal_x, normal_y = get_platform_normal(ball_rect, platform)

    for i in range(MAX_PUSH_OUT):

        x += normal_x
        y += normal_y

        ball_rect.x = round(x)
        ball_rect.y = round(y)

        if not check_platform_collision(ball_rect, ball_mask, platform):
            break

    # Velocity of the platfrom at point of contact.
    platform_vel_x, platform_vel_y = calculate_platform_velocity(
        platform,
        ball_rect
    )

    # The ball's speed compared to the platform.
    relative_vel_x = vel_x - platform_vel_x
    relative_vel_y = vel_y - platform_vel_y

    # checks to make sure the ball only bounces if it is heading into the platform.
    moving_into_platform = (
        relative_vel_x * normal_x + relative_vel_y * normal_y
    ) < 0

    if moving_into_platform:
        vel_x, vel_y = calculate_bounce(
            vel_x,
            vel_y,
            normal_x,
            normal_y,
            platform_vel_x,
            platform_vel_y
        )

    return x, y, vel_x, vel_y


def predict_trajectory(ball, platforms, prediction_time=1.0):
    # Predicts where the ball will go.

    positions = []

    # Copy the ball's position and speed so the real ball isn't touched.
    x = ball.x
    y = ball.y

    vel_x = ball.vel_x
    vel_y = ball.vel_y

    width = ball.rect.width
    height = ball.rect.height

    predicted_rect = pygame.Rect(
        round(x),
        round(y),
        width,
        height
    )

    dt = prediction_time / PREDICTION_STEPS

    for i in range(PREDICTION_STEPS):

        # Gravity.
        vel_y += GRAVITY * dt

        # Air resistance.
        vel_x, vel_y = apply_air_resistance(
            vel_x,
            vel_y,
            dt
        )

        dx = vel_x * dt
        dy = vel_y * dt

        # Bounce off the four edges of the screen.
        if x + dx < 0:
            dx = -x
            vel_x = -vel_x

        if x + width + dx > SCREEN_WIDTH:
            dx = SCREEN_WIDTH - (x + width)
            vel_x = -vel_x

        if y + dy < 0:
            dy = -y
            vel_y = -vel_y

        if y + height + dy > SCREEN_HEIGHT:
            dy = SCREEN_HEIGHT - (y + height)
            vel_y = -vel_y

        x += dx
        y += dy

        predicted_rect.x = round(x)
        predicted_rect.y = round(y)

        # Bounce off the platforms.
        for platform in platforms:

            x, y, vel_x, vel_y = handle_platform_collision(
                x,
                y,
                vel_x,
                vel_y,
                predicted_rect,
                ball.mask,
                platform
            )

        positions.append(predicted_rect.center)

    return positions


def draw_trajectory(positions):
    # Draws a dot for each predicted position.

    total = len(positions)

    for i, position in enumerate(positions):

        # How far along the path this dot is: 0 = start, 1 = end.
        progress = i / max(1, total - 1)

        if progress < 0.5:
            # First half: green fades to yellow.
            blend = progress * 2

            red = int(255 * blend)
            green = 255
            blue = 0

        else:
            # Second half: yellow fades to red.
            blend = (progress - 0.5) * 2

            red = 255
            green = int(255 * (1 - blend))
            blue = 0

        # Transparency of the dot.
        color = (red, green, blue, TRAJECTORY_ALPHA)

        pygame.draw.circle(
            screen,
            color,
            position,
            TRAJECTORY_DOT
        )


# ---------------------------------------------------------------- the ball
class Ball:

    def __init__(self, x, y):

        # The ball's picture, and a mask of which pixels are solid.
        self.image = jumpy_image
        self.mask = pygame.mask.from_surface(self.image)

        # The box around the ball, placed with its center at (x, y).
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

        self.x = float(self.rect.x)
        self.y = float(self.rect.y)

        self.vel_x = BALL_SPEED / math.sqrt(2)
        self.vel_y = -BALL_SPEED / math.sqrt(2)

    def move(self, delta_time, platforms):

        # Gravity.
        self.vel_y += GRAVITY * delta_time

        # Air resistance.
        self.vel_x, self.vel_y = apply_air_resistance(
            self.vel_x,
            self.vel_y,
            delta_time
        )

        dx = self.vel_x * delta_time
        dy = self.vel_y * delta_time

        # Bounce off the four edges of the screen.
        if self.x + dx < 0:
            dx = -self.x
            self.vel_x = -self.vel_x

        if self.x + self.rect.width + dx > SCREEN_WIDTH:
            dx = SCREEN_WIDTH - (self.x + self.rect.width)
            self.vel_x = -self.vel_x

        if self.y + dy < 0:
            dy = -self.y
            self.vel_y = -self.vel_y

        if self.y + self.rect.height + dy > SCREEN_HEIGHT:
            dy = SCREEN_HEIGHT - (self.y + self.rect.height)
            self.vel_y = -self.vel_y

        self.x += dx
        self.y += dy

        self.rect.x = round(self.x)
        self.rect.y = round(self.y)

        # Check every platform for collision.
        for platform in platforms:

            self.x, self.y, self.vel_x, self.vel_y = handle_platform_collision(
                self.x,
                self.y,
                self.vel_x,
                self.vel_y,
                self.rect,
                self.mask,
                platform
            )

    def draw(self):

        # Draw the ball's picture.
        screen.blit(self.image, self.rect)

        # Shows ball hitbox for degugging.
        if DEBUG:
            draw_mask_outline(self.mask, self.rect, WHITE)


# ------------------------------------------------------------ platform helpers
def quad_angle(quad):
    dx = quad[1][0] - quad[0][0]
    dy = quad[1][1] - quad[0][1]
    return math.degrees(math.atan2(-dy, dx))


def clamp_vec(x, y, limit):
    # If a movement (x, y) is faster than the limit, shrink it to the limit. Direction stays the same.
    mag = math.hypot(x, y)
    if mag > limit:
        return x * limit / mag, y * limit / mag
    return x, y


# ------------------------------------------------------------ the platform
class Platform(pygame.sprite.Sprite):
    # Creates the platform on the apriltag.

    def __init__(self, quad):

        super().__init__()

        self.quad = quad.copy()
        self.start_quad = quad.copy()
        self.target_quad = quad.copy()

        self.angle = quad_angle(quad)
        self.start_angle = self.angle
        self.target_angle = self.angle

        # How fast the platform is moving.
        self.vel_x = 0.0
        self.vel_y = 0.0

        # Check for spin
        self.angular_velocity = 0.0

        # The collision mask is built  when it is needed.
        self._mask = None
        self.update_bounds()

    def update_bounds(self):
        # Recalculates the platform's box and center from its four corners.

        q = self.quad
        xs = q[:, 0]
        ys = q[:, 1]
        x0 = int(math.floor(xs.min()))
        y0 = int(math.floor(ys.min()))
        x1 = int(math.ceil(xs.max()))
        y1 = int(math.ceil(ys.max()))

        # The smallest box that fits around the platform.
        self.rect = pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        # The middle of the platform.
        self.center = (float(xs.mean()), float(ys.mean()))
        # Removes old mask.
        self._mask = None

    @property
    def mask(self):
        # The collision mask.

        if self._mask is None:
            # Creates mask the shape of the perimeter of the tag.
            surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            points = [(float(x - self.rect.x), float(y - self.rect.y))
                      for x, y in self.quad]
            pygame.draw.polygon(surf, PLATFORM_COLOR, points)
            self._mask = pygame.mask.from_surface(surf)

        return self._mask

    def set_target(self, latest_quad, delta_time):
        # Called when the tag is visible. Sets where the platform should move to this frame.
        if PLATFORM_FOLLOW_TIME <= 0:
            k = 1.0
        else:
            k = 1.0 - math.exp(-delta_time / PLATFORM_FOLLOW_TIME)

        target = self.quad + (latest_quad - self.quad) * k

        self.start_quad = self.quad.copy()
        self.start_angle = self.angle
        start_center = self.center

        self.target_quad = target

        diff = (quad_angle(target) - self.angle + 180) % 360 - 180
        self.target_angle = self.angle + diff

        target_center = (float(target[:, 0].mean()), float(target[:, 1].mean()))

        raw_vx = (target_center[0] - start_center[0]) / delta_time
        raw_vy = (target_center[1] - start_center[1]) / delta_time
        raw_w = diff / delta_time

        s = PLATFORM_VEL_SMOOTH
        self.vel_x = s * raw_vx + (1 - s) * self.vel_x
        self.vel_y = s * raw_vy + (1 - s) * self.vel_y
        self.vel_x, self.vel_y = clamp_vec(self.vel_x, self.vel_y, MAX_PLATFORM_SPEED)

        self.angular_velocity = s * raw_w + (1 - s) * self.angular_velocity
        self.angular_velocity = max(-MAX_PLATFORM_ROT_SPEED,
                                    min(MAX_PLATFORM_ROT_SPEED, self.angular_velocity))

    def hold(self):
        # Called when the tag is missing for a moment. Keeps the platform still.
        self.start_quad = self.quad.copy()
        self.target_quad = self.quad.copy()
        self.start_angle = self.angle
        self.target_angle = self.angle
        self.vel_x = self.vel_y = 0.0
        self.angular_velocity = 0.0

    def move_part_way(self, fraction):
        # Moves the platform part of the way from its start to its target.

        self.quad = self.start_quad + (self.target_quad - self.start_quad) * fraction

        self.angle = (
            self.start_angle
            + (self.target_angle - self.start_angle) * fraction
        )

        self.update_bounds()

    def draw(self):
        # Draws the platform as a see-through filled shape.
        points = [(float(x), float(y)) for x, y in self.quad]
        pygame.draw.polygon(screen, PLATFORM_COLOR + (PLATFORM_ALPHA,), points)

        # In debug mode, also draw a red outline.
        if DEBUG:
            pygame.draw.lines(screen, RED, True, points, 2)


# --------------------------------------------------- camera + tag detection
class Vision:

    def __init__(self, camera, detector):
        self.camera = camera
        self.detector = detector
        # A lock stops the two threads from changing shared data at the same moment, helps performance.
        self.lock = threading.Lock()
        self.new_raw = threading.Event()
        self.running = True
        self.error = None

        self.raw = None
        self.raw_seq = 0
        self.result = None

    def start(self):
        # Start both background threads.
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _capture_loop(self):
        # Keep reading the camera. Only the newest picture is kept for performance.
        while self.running:
            ok, frame = self.camera.read()
            if not ok:
                self.error = "Cam not Working"
                break
            with self.lock:
                self.raw = frame
                self.raw_seq += 1
            self.new_raw.set()

    def _detect_loop(self):
        last_seq = 0
        while self.running:
            self.new_raw.wait(0.5)
            self.new_raw.clear()

            with self.lock:
                frame, seq = self.raw, self.raw_seq
            if frame is None or seq == last_seq:
                continue
            last_seq = seq

            small = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT),
                               interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # Find the tags. Keep each tag's id and its four corner points.
            tags = {t.tag_id: t.corners.astype(np.float64) for t in self.detector.detect(gray)}

            # Save the result for the game to pick up.
            with self.lock:
                self.result = (small, tags, time.perf_counter(), seq)

    def latest(self):
        # The game calls this to get the newest finished result (or None if there is none yet).
        with self.lock:
            return self.result


def compose(frame_bgr, tags):
    # Puts the tag outlines and the game on top of the camera picture.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Draw a red outline around every tag the camera sees.
    for corners in tags.values():
        cv2.polylines(rgb, [corners.astype(np.int32)], True, (255, 0, 0), 2)

    backdrop = pygame.image.frombuffer(rgb, (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB")
    backdrop.blit(screen, (0, 0))
    del backdrop

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------- window
# Find the screen size, then make the video window about 90% of the screen.
root = tk.Tk()
screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
root.destroy()
win_scale = min(screen_w * 0.9 / SCREEN_WIDTH, screen_h * 0.9 / SCREEN_HEIGHT)
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", int(SCREEN_WIDTH * win_scale), int(SCREEN_HEIGHT * win_scale))

# ------------------------------------------------------------------- main
# Make the ball near the top middle of the screen.
ball = Ball(SCREEN_WIDTH // 2, int(100 * SCALE))

platform_group = pygame.sprite.Group()
platforms = {}            # tag id -> Platform
latest_quad = {}          # tag id -> Newest corners we saw for it
latest_stamp = {}         # tag id -> Last time tag was detected
current_ids = set()       # Tags in last detection
last_seq = 0
frame_budget = 1.0 / GAME_FPS    # how many seconds each game frame should take

# Start the camera and tag detection in the background.
vision = Vision(camera, detector)
vision.start()

last_t = time.perf_counter()
prev_loop_start = last_t
frame_time_avg = frame_budget

# The game loop: everything inside runs once per frame while game is running.
while True:

    loop_start = time.perf_counter()

    # Keyboard and window
    key = cv2.waitKey(1) & 0xFF
    # Quit with q, or if the window was closed.
    if key == ord("q") or cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
        break
    # Space puts the ball back at the top.
    if key == ord(" "):
        ball = Ball(SCREEN_WIDTH // 2, int(100 * SCALE))

    # Stop if the camera stopped working.
    if vision.error:
        print(vision.error)
        break

    # Get the newest camera picture and tags
    result = vision.latest()
    if result is None:
        time.sleep(0.01)
        last_t = time.perf_counter()
        continue

    frame, tags, stamp, seq = result

    now = time.perf_counter()
    delta_time = max(0.001, min(0.1, now - last_t))
    last_t = now

    if seq != last_seq:
        last_seq = seq
        current_ids = set(tags)
        for tag_id, corners in tags.items():
            # Grow or shrink the tag's shape around its middle by HIT_SCALE.
            centre = corners.mean(axis=0)
            latest_quad[tag_id] = centre + (corners - centre) * HIT_SCALE
            latest_stamp[tag_id] = stamp

    # Make a platform for every newly seen tag
    for tag_id in current_ids:
        if tag_id not in platforms:
            platform = Platform(latest_quad[tag_id])
            platforms[tag_id] = platform
            platform_group.add(platform)

    # Update every platform
    for tag_id in list(platforms):
        platform = platforms[tag_id]
        if now - latest_stamp[tag_id] > PLATFORM_HOLD:
            platform_group.remove(platform)
            del platforms[tag_id]
            del latest_quad[tag_id]
            del latest_stamp[tag_id]
        elif tag_id in current_ids:
            platform.set_target(latest_quad[tag_id], delta_time)
        else:
            platform.hold()

    step_time = delta_time / SUBSTEPS

    for step in range(SUBSTEPS):

        # Move the frame with the tag.
        fraction = (step + 1) / SUBSTEPS

        for platform in platform_group:
            platform.move_part_way(fraction)

        ball.move(step_time, platform_group)

    # Work out where the ball is going next
    predicted_positions = predict_trajectory(
        ball,
        platform_group,
        PREDICTION_TIME
    )

    # Draw the game on the see-through layer
    screen.fill((0, 0, 0, 0))

    for platform in platform_group:
        platform.draw()

    draw_trajectory(predicted_positions)

    ball.draw()

    # Put the game on top of the camera picture
    out = compose(frame, {i: tags[i] for i in current_ids if i in tags})

    # ---- measure the frame rate
    frame_time_avg = 0.95 * frame_time_avg + 0.05 * (loop_start - prev_loop_start)
    prev_loop_start = loop_start
    # Write the tag count, frame rate and key help at the bottom.
    cv2.putText(out,
                f"tags {len(platforms)}   {1.0 / max(frame_time_avg, 1e-6):.0f} fps   "
                "[space] reset ball  [q] quit",
                (30, SCREEN_HEIGHT - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the finished picture
    cv2.imshow("Webcam", out)

    spare = frame_budget - (time.perf_counter() - loop_start)
    if spare > 0:
        time.sleep(spare)


# Close the game
vision.stop()
camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
pygame.quit()