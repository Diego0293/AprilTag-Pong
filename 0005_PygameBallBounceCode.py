import pygame
import math

pygame.init()

# Game window dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500

# Create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pong")

# Set frame rate
FPS = 60
clock = pygame.time.Clock()

# Game variables
GRAVITY = 500  # Pixels per second squared

# Set to True to draw the outline of each mask (helpful for checking collisions)
DEBUG = True

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Load images
bg_image = pygame.image.load("assets/background.png").convert_alpha()
bg_image = pygame.transform.scale(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

jumpy_image = pygame.image.load("assets/jumpy.png").convert_alpha()
jumpy_image = pygame.transform.scale(jumpy_image, (40,40))

platform_image = pygame.image.load("assets/platform.png").convert_alpha()


# Draws the outline of a mask on the screen (only used when DEBUG is True)
def draw_mask_outline(mask, rect, color):
    points = mask.outline()

    # Move each point from mask coordinates to screen coordinates
    screen_points = []
    for point in points:
        screen_points.append((point[0] + rect.x, point[1] + rect.y))

    # Need at least 2 points to draw a line
    if len(screen_points) > 1:
        pygame.draw.lines(screen, color, True, screen_points, 2)


# Ball class
class Ball:
    def __init__(self, x, y):
        self.image = jumpy_image

        # Mask
        self.mask = pygame.mask.from_surface(self.image)

        # The rect is the full image size so it lines up with the mask
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

        # Physics variables
        self.x = float(self.rect.x)
        self.y = float(self.rect.y)

        # Starting direction and speed (pixels per second)
        self.vel_x = 200
        self.vel_y = -200

    def move(self, delta_time, platforms):
        # Apply gravity
        self.vel_y += GRAVITY * delta_time

        # How far the ball wants to move this frame
        dx = self.vel_x * delta_time
        dy = self.vel_y * delta_time

        # Ensure ball doesn't go off the edge of the screen (bounce off walls)
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

        for platform in platforms:
            if pygame.sprite.collide_mask(self, platform):
                self.rect.x = round(self.x)
                self.rect.y = round(self.y)

                # Direction the platform is facing (straight up when angle is 0)
                angle = math.radians(platform.angle + 90)
                dir_x = math.cos(angle)
                dir_y = -math.sin(angle)  # minus because screen y goes down

                # If the ball is on the other side of the platform, use the opposite direction
                to_ball_x = self.rect.centerx - platform.rect.centerx
                to_ball_y = self.rect.centery - platform.rect.centery
                if to_ball_x * dir_x + to_ball_y * dir_y < 0:
                    dir_x = -dir_x
                    dir_y = -dir_y

                # Keep the same speed, but send it in the platform's direction
                speed = math.sqrt(self.vel_x * self.vel_x + self.vel_y * self.vel_y)
                self.vel_x = dir_x * speed
                self.vel_y = dir_y * speed

    def draw(self):
        screen.blit(self.image, self.rect)

        # Draw mask outline for debugging
        if DEBUG:
            draw_mask_outline(self.mask, self.rect, WHITE)


# Platform class
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width):
        super().__init__()

        self.original_image = pygame.transform.scale(platform_image, (width, 20))
        self.image = self.original_image

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

        self.mask = pygame.mask.from_surface(self.image)

        self.angle = 0
        self.rotation_speed = 180  # Degrees per second

    def update(self, delta_time):
        # Follow mouse
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.rect.center = (mouse_x, mouse_y)

        # Check keyboard input
        key = pygame.key.get_pressed()

        if key[pygame.K_q]:
            self.angle += self.rotation_speed * delta_time

        if key[pygame.K_e]:
            self.angle -= self.rotation_speed * delta_time

        # Rotate image
        self.image = pygame.transform.rotate(self.original_image, self.angle)

        # Update collision rectangle while preserving center
        self.rect = self.image.get_rect(center=self.rect.center)

        # Rebuild the mask, since the rotated image has a new shape
        self.mask = pygame.mask.from_surface(self.image)


# Create ball instance
ball = Ball(SCREEN_WIDTH // 2, 100)

# Create platform instances
platform_group = pygame.sprite.Group(Platform(0, 0, 150))


# Main game loop
running = True

while running:
    delta_time = clock.tick(FPS) / 1000                 # Calculate elapsed time in seconds
    delta_time = max(0.001, min(0.1, delta_time))       # Limit unusually large time steps

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update platforms FIRST so their masks are up to date
    platform_group.update(delta_time)

    # Then move the ball and check collisions against the platforms
    ball.move(delta_time, platform_group)

    # Draw background
    screen.fill((118, 171, 251))
    screen.blit(bg_image, (0, 0))

    # Draw platforms
    platform_group.draw(screen)
    if DEBUG:
        for platform in platform_group:
            draw_mask_outline(platform.mask, platform.rect, RED)

    # Draw ball
    ball.draw()

    # Update display
    pygame.display.flip()

pygame.quit()