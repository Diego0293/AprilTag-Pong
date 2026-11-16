import pygame
import math

pygame.init()

# Game window dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500

# Create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pryor to Your Understanding: The Game")

# Set frame rate
FPS = 60
clock = pygame.time.Clock()

# Game variables
GRAVITY = 1000
PLATFORM_INFLUENCE = 0.3
BALL_SPEED = 400
MAX_BALL_SPEED = 1500
AIR_RESISTANCE = 0.3
PLATFORM_ROT_SPEED = 360 * 2

# Collision settings
# Each frame is split into this many small steps so a fast paddle
# can't jump past the ball in a single frame
SUBSTEPS = 20

# Max pixels to push the ball out of the paddle
MAX_PUSH_OUT = 60

# Prediction settings
PREDICTION_TIME = 1.0
PREDICTION_STEPS = 120

# How see-through the prediction line is (0 = invisible, 255 = solid)
TRAJECTORY_ALPHA = 110

# Debug settings
DEBUG = True

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Load images
bg_image = pygame.image.load("assets/background.png").convert_alpha()
bg_image = pygame.transform.scale(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

jumpy_image = pygame.image.load("assets/jumpy.png").convert_alpha()
jumpy_image = pygame.transform.scale(jumpy_image, (40, 40))

platform_image = pygame.image.load("assets/platform.png").convert_alpha()

# Transparent surface for drawing the prediction line
trajectory_surface = pygame.Surface(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    pygame.SRCALPHA
)


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
    rx = ball_rect.centerx - platform.rect.centerx
    ry = ball_rect.centery - platform.rect.centery

    # Convert angular velocity to radians per second
    omega = math.radians(platform.angular_velocity)

    # Rotational velocity
    rotational_vel_x = -omega * ry
    rotational_vel_y = omega * rx

    # Combine linear and rotational velocity
    platform_vel_x = platform.vel_x + rotational_vel_x
    platform_vel_y = platform.vel_y + rotational_vel_y

    return platform_vel_x, platform_vel_y


# Get the direction pointing out of the platform toward the ball
def get_platform_normal(ball_rect, platform):

    angle = math.radians(platform.angle + 90)

    normal_x = math.cos(angle)
    normal_y = -math.sin(angle)

    # Direction from platform to ball
    to_ball_x = ball_rect.centerx - platform.rect.centerx
    to_ball_y = ball_rect.centery - platform.rect.centery

    # Flip the normal if it points away from the ball
    if to_ball_x * normal_x + to_ball_y * normal_y < 0:
        normal_x = -normal_x
        normal_y = -normal_y

    return normal_x, normal_y


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
            2
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


# Platform class
class Platform(pygame.sprite.Sprite):

    def __init__(self, x, y, width):

        super().__init__()

        self.original_image = pygame.transform.scale(
            platform_image,
            (width, 20)
        )

        self.image = self.original_image

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

        self.mask = pygame.mask.from_surface(self.image)

        self.angle = 0
        self.rotation_speed = PLATFORM_ROT_SPEED

        # Linear velocity
        self.vel_x = 0
        self.vel_y = 0

        # Angular velocity (degrees per second)
        self.angular_velocity = 0

        # Where the platform starts and ends this frame
        self.start_x = x
        self.start_y = y
        self.start_angle = 0

        self.target_x = x
        self.target_y = y
        self.target_angle = 0

    def update(self, delta_time):

        # Where the platform is now
        self.start_x = self.rect.centerx
        self.start_y = self.rect.centery
        self.start_angle = self.angle

        # Where the platform should be at the end of this frame
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.target_x = mouse_x
        self.target_y = mouse_y

        self.target_angle = self.angle

        # Keyboard input
        key = pygame.key.get_pressed()

        if key[pygame.K_q]:
            self.target_angle += self.rotation_speed * delta_time

        if key[pygame.K_e]:
            self.target_angle -= self.rotation_speed * delta_time

        # Calculate linear velocity
        self.vel_x = (self.target_x - self.start_x) / delta_time
        self.vel_y = (self.target_y - self.start_y) / delta_time

        # Calculate angular velocity
        self.angular_velocity = (
            self.target_angle - self.start_angle
        ) / delta_time

    # Move part of the way from start to target
    # fraction = 0 is the start, fraction = 1 is the target
    def move_part_way(self, fraction):

        x = self.start_x + (self.target_x - self.start_x) * fraction
        y = self.start_y + (self.target_y - self.start_y) * fraction

        self.angle = (
            self.start_angle
            + (self.target_angle - self.start_angle) * fraction
        )

        # Rotate image
        self.image = pygame.transform.rotate(
            self.original_image,
            self.angle
        )

        # Keep the center in the right place
        self.rect = self.image.get_rect(
            center=(round(x), round(y))
        )

        # Rebuild rotated mask
        self.mask = pygame.mask.from_surface(self.image)


# Create ball
ball = Ball(SCREEN_WIDTH // 2, 100)

# Create platforms
platform_group = pygame.sprite.Group(
    Platform(0, 0, 150)
)


# Main game loop
running = True

while running:

    delta_time = clock.tick(FPS) / 1000
    delta_time = max(0.001, min(0.1, delta_time))

    # Handle events
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    # Work out where platforms are going this frame
    platform_group.update(delta_time)

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

    # Draw background
    screen.fill((118, 171, 251))
    screen.blit(bg_image, (0, 0))

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

    # Update display
    pygame.display.flip()


pygame.quit()