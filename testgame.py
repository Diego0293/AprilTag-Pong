import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import math
import time
import tkinter as tk

import cv2
import numpy as np
import pygame
from pupil_apriltags import Detector

# ----------------------------------------------------------------- settings
GAME_W = 1280             # pygame surface width; height follows the camera aspect
HIT_SCALE = 1.0           # grow/shrink each platform around its tag (1.0 = exactly the tag)
PLATFORM_HOLD = 0.25      # seconds a platform survives after its tag is lost
PLATFORM_VEL_SMOOTH = 0.5 # 1 = raw tag motion, lower = smoother (less jitter)
DEBUG = True              # draw mask outlines like your original

BASE_GRAVITY = 1000
PLATFORM_INFLUENCE = 0.3
BASE_BALL_SPEED = 400
BASE_MAX_BALL_SPEED = 1500
AIR_RESISTANCE = 0.3
BASE_MAX_PLATFORM_SPEED = 4000
MAX_PLATFORM_ROT_SPEED = 2000     # degrees per second

SUBSTEPS = 20
BASE_MAX_PUSH_OUT = 60

PREDICTION_TIME = 1.0
PREDICTION_STEPS = 120
TRAJECTORY_ALPHA = 110

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PLATFORM_COLOR = (80, 220, 255)
PLATFORM_ALPHA = 140

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

num_cores = os.cpu_count() or 1
cv2.setNumThreads(num_cores)
print("Using", num_cores, "cores")
detector = Detector(families="tag36h11", nthreads=num_cores)

# ------------------------------------------------------------------- camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
camera.set(cv2.CAP_PROP_FPS, 30)

cam_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera resolution:", cam_w, "x", cam_h)

SCREEN_WIDTH = GAME_W
SCREEN_HEIGHT = int(GAME_W * cam_h / cam_w)
SCALE = SCREEN_HEIGHT / 500.0
TO_GAME = SCREEN_WIDTH / cam_w

GRAVITY = BASE_GRAVITY * SCALE
BALL_SPEED = BASE_BALL_SPEED * SCALE
MAX_BALL_SPEED = BASE_MAX_BALL_SPEED * SCALE
MAX_PLATFORM_SPEED = BASE_MAX_PLATFORM_SPEED * SCALE
MAX_PUSH_OUT = int(BASE_MAX_PUSH_OUT * SCALE)
BALL_SIZE = int(40 * SCALE)
TRAJECTORY_DOT = max(2, int(2 * SCALE))

# ------------------------------------------------------------------- pygame
pygame.display.init()
pygame.display.set_mode((1, 1))

# Transparent game layer: whatever isn't drawn shows the camera underneath
screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

# Transparent surface for drawing the prediction line
trajectory_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)


def make_fallback_ball():
    surf = pygame.Surface((BALL_SIZE, BALL_SIZE), pygame.SRCALPHA)
    c = BALL_SIZE // 2
    pygame.draw.circle(surf, (255, 200, 0), (c, c), c)
    pygame.draw.circle(surf, (40, 40, 40), (c, c), c, max(2, BALL_SIZE // 12))
    return surf


def load_ball_image():
    path = os.path.join(ASSET_DIR, "jumpy.png")
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(img, (BALL_SIZE, BALL_SIZE))
    except (FileNotFoundError, pygame.error):
        print("assets/jumpy.png not found, drawing a plain ball instead")
        return make_fallback_ball()


jumpy_image = load_ball_image()


# Draw mask outline
def draw_mask_outline(mask, rect, color):

    points = mask.outline()
    screen_points = []

    for point in points:
        screen_points.append(
            (point[0] + rect.x, point[1] + rect.y)
        )

    if len(screen_points) > 1:
        pygame.draw.lines(screen, color, True, screen_points, 2)


# Check collision between ball and platform masks
def check_platform_collision(ball_rect, ball_mask, platform):

    offset = (
        ball_rect.x - platform.rect.x,
        ball_rect.y - platform.rect.y
    )

    return platform.mask.overlap(ball_mask, offset) is not None


# Apply air resistance while maintaining minimum and maximum speed
def apply_air_resistance(vel_x, vel_y, delta_time):

    speed = math.sqrt(vel_x ** 2 + vel_y ** 2)

    if speed == 0:
        return BALL_SPEED / math.sqrt(2), -BALL_SPEED / math.sqrt(2)

    # Apply air resistance
    drag_factor = max(0, 1 - AIR_RESISTANCE * delta_time)
    new_speed = speed * drag_factor

    # Enforce minimum and maximum speed
    new_speed = max(BALL_SPEED, min(new_speed, MAX_BALL_SPEED))

    # Preserve velocity direction
    scale = new_speed / speed

    vel_x *= scale
    vel_y *= scale

    return vel_x, vel_y


# Calculate platform velocity at the ball's position
def calculate_platform_velocity(platform, ball_rect):

    # Distance from platform center to ball center
    rx = ball_rect.centerx - platform.center[0]
    ry = ball_rect.centery - platform.center[1]

    # Convert angular velocity to radians per second
    # (platform angles are counter-clockwise as seen on screen, y points down)
    omega = math.radians(platform.angular_velocity)

    # Rotational velocity
    rotational_vel_x = omega * ry
    rotational_vel_y = -omega * rx

    # Combine linear and rotational velocity
    platform_vel_x = platform.vel_x + rotational_vel_x
    platform_vel_y = platform.vel_y + rotational_vel_y

    return platform_vel_x, platform_vel_y


# Get the direction pointing out of the platform toward the ball.
# Uses the tag's real outline: the normal of the nearest edge (or the
# direction from the nearest outline point to the ball).
def get_platform_normal(ball_rect, platform):

    px, py = ball_rect.center
    cx, cy = platform.center
    quad = platform.quad

    best = None
    inside = True

    for i in range(4):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % 4]
        abx, aby = bx - ax, by - ay
        length_sq = abx * abx + aby * aby

        if length_sq < 1e-9:
            continue

        length = math.sqrt(length_sq)

        # Edge normal, flipped to point away from the platform centre
        nx, ny = aby / length, -abx / length
        mx, my = (ax + bx) / 2, (ay + by) / 2
        if nx * (mx - cx) + ny * (my - cy) < 0:
            nx, ny = -nx, -ny

        if (px - ax) * nx + (py - ay) * ny > 0:
            inside = False

        # Closest point on this edge to the ball
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
        qx, qy = ax + t * abx, ay + t * aby
        dist = math.hypot(px - qx, py - qy)

        if best is None or dist < best[0]:
            best = (dist, qx, qy, nx, ny)

    if best is None:
        return 0.0, -1.0

    dist, qx, qy, nx, ny = best

    if inside or dist < 1e-6:
        return nx, ny

    return (px - qx) / dist, (py - qy) / dist


# Calculate bounce direction and velocity
def calculate_bounce(vel_x, vel_y, normal_x, normal_y,
                     platform_vel_x, platform_vel_y):

    # Calculate relative ball velocity
    relative_vel_x = vel_x - platform_vel_x
    relative_vel_y = vel_y - platform_vel_y

    # Preserve relative speed
    speed = math.sqrt(relative_vel_x ** 2 + relative_vel_y ** 2)

    # Apply bounce direction and platform velocity
    vel_x = normal_x * speed + platform_vel_x * PLATFORM_INFLUENCE
    vel_y = normal_y * speed + platform_vel_y * PLATFORM_INFLUENCE

    return vel_x, vel_y


# Check one platform, push the ball out of it and bounce if needed
def handle_platform_collision(x, y, vel_x, vel_y,
                              ball_rect, ball_mask, platform):

    # No overlap, nothing to do
    if not check_platform_collision(ball_rect, ball_mask, platform):
        return x, y, vel_x, vel_y

    normal_x, normal_y = get_platform_normal(ball_rect, platform)

    # Push the ball out of the platform one pixel at a time
    for i in range(MAX_PUSH_OUT):

        x += normal_x
        y += normal_y

        ball_rect.x = round(x)
        ball_rect.y = round(y)

        if not check_platform_collision(ball_rect, ball_mask, platform):
            break

    # Get platform velocity at collision point
    platform_vel_x, platform_vel_y = calculate_platform_velocity(
        platform,
        ball_rect
    )

    # Ball velocity compared to the platform
    relative_vel_x = vel_x - platform_vel_x
    relative_vel_y = vel_y - platform_vel_y

    # Only bounce if the ball is moving into the platform
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


# Predict future trajectory
def predict_trajectory(ball, platforms, prediction_time=1.0):

    positions = []

    # Copy ball physics
    x = ball.x
    y = ball.y

    vel_x = ball.vel_x
    vel_y = ball.vel_y

    width = ball.rect.width
    height = ball.rect.height

    # Independent predicted rectangle
    predicted_rect = pygame.Rect(
        round(x),
        round(y),
        width,
        height
    )

    dt = prediction_time / PREDICTION_STEPS

    for i in range(PREDICTION_STEPS):

        # Apply gravity
        vel_y += GRAVITY * dt

        # Apply air resistance
        vel_x, vel_y = apply_air_resistance(
            vel_x,
            vel_y,
            dt
        )

        # Calculate movement
        dx = vel_x * dt
        dy = vel_y * dt

        # Wall collisions
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

        # Update predicted position
        x += dx
        y += dy

        predicted_rect.x = round(x)
        predicted_rect.y = round(y)

        # Check platform collisions
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


# Draw predicted trajectory with color gradient
def draw_trajectory(positions):

    # Clear last frame's dots
    trajectory_surface.fill((0, 0, 0, 0))

    total = len(positions)

    for i, position in enumerate(positions):

        progress = i / max(1, total - 1)

        if progress < 0.5:
            # Green to yellow
            blend = progress * 2

            red = int(255 * blend)
            green = 255
            blue = 0

        else:
            # Yellow to red
            blend = (progress - 0.5) * 2

            red = 255
            green = int(255 * (1 - blend))
            blue = 0

        # Fourth number is transparency
        color = (red, green, blue, TRAJECTORY_ALPHA)

        pygame.draw.circle(
            trajectory_surface,
            color,
            position,
            TRAJECTORY_DOT
        )

    # Draw the see-through layer on top of the screen
    screen.blit(trajectory_surface, (0, 0))


# Ball class
class Ball:

    def __init__(self, x, y):

        self.image = jumpy_image
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

        # Physics variables
        self.x = float(self.rect.x)
        self.y = float(self.rect.y)

        # Initial total speed is BALL_SPEED
        self.vel_x = BALL_SPEED / math.sqrt(2)
        self.vel_y = -BALL_SPEED / math.sqrt(2)

    def move(self, delta_time, platforms):

        # Apply gravity
        self.vel_y += GRAVITY * delta_time

        # Apply air resistance
        self.vel_x, self.vel_y = apply_air_resistance(
            self.vel_x,
            self.vel_y,
            delta_time
        )

        dx = self.vel_x * delta_time
        dy = self.vel_y * delta_time

        # Wall collisions
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

        # Update position
        self.x += dx
        self.y += dy

        self.rect.x = round(self.x)
        self.rect.y = round(self.y)

        # Platform collisions
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

        screen.blit(self.image, self.rect)

        if DEBUG:
            draw_mask_outline(self.mask, self.rect, WHITE)


def quad_angle(quad):
    """Tag rotation in degrees, counter-clockwise as seen on screen."""
    dx = quad[1][0] - quad[0][0]
    dy = quad[1][1] - quad[0][1]
    return math.degrees(math.atan2(-dy, dx))


def clamp_vec(x, y, limit):
    mag = math.hypot(x, y)
    if mag > limit:
        return x * limit / mag, y * limit / mag
    return x, y


# Platform class: one per AprilTag
class Platform(pygame.sprite.Sprite):

    def __init__(self, quad):

        super().__init__()

        self.quad = quad.copy()
        self.start_quad = quad.copy()
        self.target_quad = quad.copy()

        self.angle = quad_angle(quad)
        self.start_angle = self.angle
        self.target_angle = self.angle

        # Linear velocity
        self.vel_x = 0
        self.vel_y = 0

        # Angular velocity (degrees per second)
        self.angular_velocity = 0

        self.rebuild()

    # Rebuild the rect / mask from the current outline of the tag
    def rebuild(self):

        q = self.quad
        x0 = int(math.floor(q[:, 0].min()))
        y0 = int(math.floor(q[:, 1].min()))
        x1 = int(math.ceil(q[:, 0].max()))
        y1 = int(math.ceil(q[:, 1].max()))
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)

        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        points = [(float(x - x0), float(y - y0)) for x, y in q]
        pygame.draw.polygon(surf, PLATFORM_COLOR, points)

        self.mask = pygame.mask.from_surface(surf)
        self.image = surf
        self.image.set_alpha(PLATFORM_ALPHA)
        self.rect = pygame.Rect(x0, y0, w, h)
        self.center = (float(q[:, 0].mean()), float(q[:, 1].mean()))

    # Tag seen this frame: where the platform starts and where it should end
    def set_target(self, quad, delta_time):

        self.start_quad = self.quad.copy()
        self.start_angle = self.angle
        start_center = self.center

        self.target_quad = quad

        # Unwrap so a tag turning past 180 degrees doesn't spin the platform
        diff = (quad_angle(quad) - self.angle + 180) % 360 - 180
        self.target_angle = self.angle + diff

        target_center = (float(quad[:, 0].mean()), float(quad[:, 1].mean()))

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

    # Tag briefly lost: stay where it was, with no velocity
    def hold(self):

        self.start_quad = self.quad.copy()
        self.target_quad = self.quad.copy()
        self.start_angle = self.angle
        self.target_angle = self.angle
        self.vel_x = self.vel_y = 0
        self.angular_velocity = 0

    # Move part of the way from start to target
    # fraction = 0 is the start, fraction = 1 is the target
    def move_part_way(self, fraction):

        self.quad = self.start_quad + (self.target_quad - self.start_quad) * fraction

        self.angle = (
            self.start_angle
            + (self.target_angle - self.start_angle) * fraction
        )

        self.rebuild()


def overlay_game(frame, game_surface):
    """Alpha-composite the transparent game layer over the camera frame (in place)."""
    rgb = pygame.surfarray.array3d(game_surface).transpose(1, 0, 2)    # (h, w, 3) RGB
    alpha = pygame.surfarray.array_alpha(game_surface).T                # (h, w)

    # Premultiply (BGR order) so resizing doesn't bleed black into the edges
    premult = (rgb[..., ::-1].astype(np.float32) * (alpha[..., None] / 255.0)).astype(np.uint8)

    fh, fw = frame.shape[:2]
    big_p = cv2.resize(premult, (fw, fh), interpolation=cv2.INTER_LINEAR)
    big_a = cv2.resize(np.ascontiguousarray(alpha), (fw, fh), interpolation=cv2.INTER_LINEAR)

    mask = big_a > 0
    a = (big_a[mask].astype(np.float32) / 255.0)[:, None]
    frame[mask] = np.clip(big_p[mask] + frame[mask] * (1.0 - a), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- window
text_scale = cam_w / 1280
text_thick = max(1, int(2 * text_scale))
line_thick = max(1, int(3 * text_scale))

root = tk.Tk()
screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
root.destroy()
win_scale = min(screen_w * 0.9 / cam_w, screen_h * 0.9 / cam_h)
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", int(cam_w * win_scale), int(cam_h * win_scale))

# -------------------------------------------------------------------- main
ball = Ball(SCREEN_WIDTH // 2, int(100 * SCALE))

platform_group = pygame.sprite.Group()
platforms = {}            # tag_id -> Platform
last_seen = {}            # tag_id -> time it was last detected
last_t = time.perf_counter()

running = True

while running:

    key = cv2.waitKey(1) & 0xFF
    ok, frame = camera.read()
    if not ok:
        print("Cam not Working :(")
        break
    if key == ord("q") or cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord(" "):
        ball = Ball(SCREEN_WIDTH // 2, int(100 * SCALE))

    now = time.perf_counter()
    delta_time = max(0.001, min(0.1, now - last_t))
    last_t = now

    # ---- find the tags
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    seen_now = set()

    for tag in detections:

        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]),
                     (0, 0, 255), line_thick)

        quad = tag.corners.astype(np.float64) * TO_GAME
        centre = quad.mean(axis=0)
        quad = centre + (quad - centre) * HIT_SCALE

        platform = platforms.get(tag.tag_id)
        if platform is None:
            platform = Platform(quad)
            platforms[tag.tag_id] = platform
            platform_group.add(platform)
        else:
            platform.set_target(quad, delta_time)

        last_seen[tag.tag_id] = now
        seen_now.add(tag.tag_id)

    # ---- tags that vanished: hold briefly, then remove
    for tag_id in list(platforms):
        if tag_id in seen_now:
            continue
        if now - last_seen[tag_id] > PLATFORM_HOLD:
            platform_group.remove(platforms.pop(tag_id))
            del last_seen[tag_id]
        else:
            platforms[tag_id].hold()

    # Move platforms and ball together in small steps
    step_time = delta_time / SUBSTEPS

    for step in range(SUBSTEPS):

        fraction = (step + 1) / SUBSTEPS

        for platform in platform_group:
            platform.move_part_way(fraction)

        ball.move(step_time, platform_group)

    # Predict trajectory
    predicted_positions = predict_trajectory(
        ball,
        platform_group,
        PREDICTION_TIME
    )

    # Draw the game layer (transparent background)
    screen.fill((0, 0, 0, 0))

    # Draw platforms
    platform_group.draw(screen)

    if DEBUG:
        for platform in platform_group:
            draw_mask_outline(
                platform.mask,
                platform.rect,
                RED
            )

    # Draw predicted trajectory
    draw_trajectory(predicted_positions)

    # Draw ball
    ball.draw()

    # Put the game on top of the camera feed
    overlay_game(frame, screen)

    cv2.putText(frame, f"tags {len(platforms)}   [space] reset ball  [q] quit",
                (int(40 * text_scale), cam_h - int(30 * text_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * text_scale, (255, 255, 255), text_thick)

    cv2.imshow("Webcam", frame)


camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
pygame.quit()