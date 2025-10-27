import pygame
import numpy as np
import math
import time
from termcolor import colored

# ==================== CONFIGURATION ====================
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 60

# Arena (centered left side)
ARENA_SIZE = 600
ARENA_X = 100
ARENA_Y = 150

# Mouse
MOUSE_SIZE = 20
MOUSE_SPEED = 3
MOUSE_START_X = ARENA_SIZE // 2
MOUSE_START_Y = ARENA_SIZE // 2
MOUSE_COLOR = (180, 160, 140)
MOUSE_EAR_COLOR = (200, 180, 160)
MOUSE_NOSE_COLOR = (255, 200, 200)

# Food (Cheese)
FOOD_SIZE = 18
CHEESE_COLOR = (255, 220, 100)
CHEESE_HOLE_COLOR = (230, 190, 60)
FOOD_REWARD_DISTANCE = 35

# Unstuck mechanism
STUCK_THRESHOLD = 60  # Steps without progress
STUCK_DISTANCE_THRESHOLD = 5  # Pixels of movement to count as progress

# Network
RING_NEURONS = 36  # 10 degrees per neuron
MOTOR_NEURONS = 3  # Left, Straight, Right
LEARNING_RATE = 0.05
REWARD_DECAY = 0.95
TURN_ANGLE = 5  # degrees
INITIAL_EXPLORATION_RATE = 0.20  # 20% random actions at start
MIN_EXPLORATION_RATE = 0.05  # 5% minimum random actions
EXPLORATION_DECAY_RATE = 0.98  # Decay per episode when doing well

# Visualization positions (centered right side)
RING_CENTER_X = 1200
RING_CENTER_Y = 280
RING_RADIUS = 130
MOTOR_VIS_X = 1200
MOTOR_VIS_Y = 520  # Moved further down to avoid text overlap

# Learning plots (bottom center)
PLOT_X = 200
PLOT_Y = 850
PLOT_WIDTH = 400
PLOT_HEIGHT = 180

# Colors - Sci-Fi Theme
BG_COLOR = (10, 10, 20)
ARENA_COLOR = (20, 25, 35)
GRID_COLOR = (30, 40, 60)
TEXT_COLOR = (180, 200, 230)
RING_COLOR = (60, 80, 120)
ACTIVE_NEURON_COLOR = (0, 255, 200)
MOTOR_COLORS = [(255, 80, 120), (80, 255, 150), (100, 180, 255)]
ACCENT_COLOR = (0, 255, 200)  # Cyan sci-fi accent
GLOW_COLOR = (0, 200, 255)  # Blue glow


# ==================== LEARNING TRACKER ====================
class LearningTracker:
    """Tracks learning metrics over time for visualization"""
    
    def __init__(self, max_history=50):
        self.max_history = max_history
        self.episode_steps = []
        self.episode_rewards = []
        self.avg_steps_history = []
        
    def log_episode(self, steps, reward):
        """Log an episode's metrics"""
        self.episode_steps.append(steps)
        self.episode_rewards.append(reward)
        
        # Calculate moving average of steps (efficiency metric)
        window = min(5, len(self.episode_steps))
        avg_steps = sum(self.episode_steps[-window:]) / window
        self.avg_steps_history.append(avg_steps)
        
        # Keep only recent history
        if len(self.episode_steps) > self.max_history:
            self.episode_steps.pop(0)
            self.episode_rewards.pop(0)
            self.avg_steps_history.pop(0)
    
    def get_recent_avg_steps(self, window=10):
        """Get average steps over recent episodes"""
        if not self.episode_steps:
            return 0
        recent = self.episode_steps[-window:]
        return sum(recent) / len(recent)


# ==================== CAN RING NETWORK ====================
class RingNetwork:
    """Continuous Attractor Network - Ring topology for direction encoding"""
    
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.activity = np.zeros(num_neurons)
        
        # Create ring topology with recurrent connections
        self.weights = self._create_ring_weights()
        
    def _create_ring_weights(self):
        """Create recurrent weights that form a ring attractor"""
        weights = np.zeros((self.num_neurons, self.num_neurons))
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                # Distance on the ring (circular)
                dist = min(abs(i - j), self.num_neurons - abs(i - j))
                # Gaussian connectivity: nearby neurons excite, distant inhibit slightly
                # Reduced recurrent strength to prevent symmetric activation
                weights[i, j] = 0.5 * np.exp(-dist**2 / 6) - 0.05
                
        return weights
    
    def update(self, relative_angle, food_distance):
        """Update ring activity based on RELATIVE food direction (to mouse heading)"""
        # Convert relative angle to neuron index
        target_neuron = int((relative_angle % 360) / (360 / self.num_neurons))
        
        # Create input bump at food direction
        input_signal = np.zeros(self.num_neurons)
        strength = max(0.4, 1.0 - food_distance / 600)  # Closer = stronger
        
        for i in range(self.num_neurons):
            dist = min(abs(i - target_neuron), self.num_neurons - abs(i - target_neuron))
            input_signal[i] = strength * np.exp(-dist**2 / 2.5)
        
        # Recurrent dynamics: mostly input-driven to prevent symmetric patterns
        # Very weak recurrent influence
        self.activity = np.tanh(0.2 * self.weights @ self.activity + 1.8 * input_signal)
        
        return self.activity


# ==================== MOTOR OUTPUT NETWORK ====================
class MotorNetwork:
    """Learns to map ring activity to motor commands via reward-modulated Hebbian learning"""
    
    def __init__(self, input_size, num_motors):
        self.num_motors = num_motors
        # Initialize with small random weights
        self.weights = np.random.randn(num_motors, input_size) * 0.1
        self.activity = np.zeros(num_motors)
        self.last_input = None
        
    def forward(self, ring_activity):
        """Compute motor output from ring activity"""
        self.last_input = ring_activity.copy()
        self.activity = self.weights @ ring_activity
        return self.activity
    
    def learn(self, reward):
        """Reward-modulated Hebbian learning"""
        if self.last_input is not None and reward > 0:
            # Δw = learning_rate × reward × (input × output)
            for i in range(self.num_motors):
                self.weights[i] += LEARNING_RATE * reward * self.last_input * self.activity[i]
            
            # Normalize to prevent weights from exploding
            self.weights = np.clip(self.weights, -2, 2)


# ==================== MOUSE AGENT ====================
class Mouse:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0  # Heading angle in degrees
        self.velocity = MOUSE_SPEED
        
        # Networks
        self.ring_network = RingNetwork(RING_NEURONS)
        self.motor_network = MotorNetwork(RING_NEURONS, MOTOR_NEURONS)
        
        # Learning stats
        self.total_reward = 0
        self.episode = 0
        self.previous_distance = None
        
        # Adaptive exploration
        self.exploration_rate = INITIAL_EXPLORATION_RATE
        self.baseline_steps = None  # For tracking learning progress
        
        # Unstuck mechanism
        self.stuck_counter = 0
        self.last_position = (x, y)
        self.position_history = []
        
    def sense_food(self, food_x, food_y):
        """Calculate angle and distance to food"""
        dx = food_x - self.x
        dy = food_y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.degrees(math.atan2(dy, dx))
        return angle, distance
    
    def think_and_act(self, food_x, food_y):
        """Process sensory input through networks and take action"""
        # Check if stuck (hasn't moved much recently)
        distance_moved = math.sqrt((self.x - self.last_position[0])**2 + (self.y - self.last_position[1])**2)
        
        if distance_moved < STUCK_DISTANCE_THRESHOLD:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = (self.x, self.y)
        
        # Unstuck behavior: if stuck for too long, do random sharp turn and reverse
        if self.stuck_counter > STUCK_THRESHOLD:
            self.angle += np.random.choice([90, -90, 180])  # Sharp turn or reverse
            self.stuck_counter = 0
            print(colored("  [Unstuck mechanism activated!]", "yellow"))
        
        # Sense food (absolute angle)
        food_angle, food_distance = self.sense_food(food_x, food_y)
        
        # Calculate RELATIVE angle (food direction relative to mouse heading)
        relative_angle = (food_angle - self.angle) % 360
        
        # Update ring network with relative angle
        ring_activity = self.ring_network.update(relative_angle, food_distance)
        
        # Get motor command
        motor_output = self.motor_network.forward(ring_activity)
        
        # Adaptive epsilon-greedy exploration: decreases as mouse learns
        if np.random.random() < self.exploration_rate:
            action = np.random.randint(0, MOTOR_NEURONS)  # Random action
        else:
            # Decode motor output: [left, straight, right]
            action = np.argmax(motor_output)
        
        # Execute action
        if action == 0:  # Turn left
            self.angle -= TURN_ANGLE
        elif action == 2:  # Turn right
            self.angle += TURN_ANGLE
        # action == 1: go straight (no turn)
        
        # Move forward
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.velocity
        self.y += math.sin(rad) * self.velocity
        
        # Keep in bounds
        self.x = max(MOUSE_SIZE, min(ARENA_SIZE - MOUSE_SIZE, self.x))
        self.y = max(MOUSE_SIZE, min(ARENA_SIZE - MOUSE_SIZE, self.y))
        
        # Calculate reward (got closer to food)
        current_distance = food_distance
        reward = 0
        
        if self.previous_distance is not None:
            if current_distance < self.previous_distance:
                reward = 0.1  # Small reward for getting closer
            else:
                reward = -0.01  # Small penalty for moving away
        
        self.previous_distance = current_distance
        
        # Learn from reward
        if reward > 0:
            self.motor_network.learn(reward)
            self.total_reward += reward
        
        return current_distance
    
    def reset_position(self, steps_taken):
        """Reset mouse to center after reaching food"""
        self.x = MOUSE_START_X
        self.y = MOUSE_START_Y
        self.angle = np.random.randint(0, 360)
        self.previous_distance = None
        self.episode += 1
        
        # Adaptive exploration: reduce randomness as mouse learns
        if self.baseline_steps is None:
            # First episode: set baseline
            self.baseline_steps = steps_taken
        else:
            # If performance improved significantly, reduce exploration
            improvement = (self.baseline_steps - steps_taken) / self.baseline_steps
            
            if improvement > 0.1:  # More than 10% improvement
                # Decay exploration rate
                self.exploration_rate = max(MIN_EXPLORATION_RATE, 
                                           self.exploration_rate * EXPLORATION_DECAY_RATE)
                # Update baseline to recent performance
                self.baseline_steps = 0.9 * self.baseline_steps + 0.1 * steps_taken
        
        # Reset stuck mechanism
        self.stuck_counter = 0
        self.last_position = (self.x, self.y)


# ==================== SLIDER UI ====================
class Slider:
    """Simple slider for speed control"""
    
    def __init__(self, x, y, width, min_val, max_val, initial_val, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = 10
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        
    def handle_event(self, event):
        """Handle mouse events for slider"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            # Check if click is on slider
            if (self.x <= mouse_x <= self.x + self.width and 
                self.y - 10 <= mouse_y <= self.y + self.height + 10):
                self.dragging = True
                self._update_value(mouse_x)
                
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x, _ = event.pos
                self._update_value(mouse_x)
    
    def _update_value(self, mouse_x):
        """Update value based on mouse position"""
        relative_x = max(0, min(mouse_x - self.x, self.width))
        ratio = relative_x / self.width
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
    
    def draw(self, screen, font):
        """Draw the slider"""
        # Label
        label_text = font.render(f"{self.label}: {self.value:.1f}x", True, TEXT_COLOR)
        screen.blit(label_text, (self.x, self.y - 30))
        
        # Slider track
        pygame.draw.rect(screen, (80, 80, 100), 
                        (self.x, self.y, self.width, self.height))
        
        # Slider handle
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.x + int(ratio * self.width)
        pygame.draw.circle(screen, (150, 200, 255), 
                          (handle_x, self.y + self.height // 2), 12)
        pygame.draw.circle(screen, (200, 230, 255), 
                          (handle_x, self.y + self.height // 2), 8)


# ==================== VISUALIZATION ====================
def draw_mouse(screen, mouse):
    """Draw the mouse as an actual mouse with ears, body, and tail"""
    mx = int(ARENA_X + mouse.x)
    my = int(ARENA_Y + mouse.y)
    rad = math.radians(mouse.angle)
    
    # Tail (behind the mouse)
    tail_length = 25
    tail_segments = 6
    for i in range(tail_segments):
        t_angle = rad + math.pi + math.sin(i * 0.5) * 0.3
        t_start_x = mx - math.cos(rad) * 15 - math.cos(t_angle - 0.1) * (i * tail_length / tail_segments)
        t_start_y = my - math.sin(rad) * 15 - math.sin(t_angle - 0.1) * (i * tail_length / tail_segments)
        t_end_x = t_start_x - math.cos(t_angle) * (tail_length / tail_segments)
        t_end_y = t_start_y - math.sin(t_angle) * (tail_length / tail_segments)
        pygame.draw.line(screen, (150, 130, 110), 
                        (int(t_start_x), int(t_start_y)), 
                        (int(t_end_x), int(t_end_y)), 3)
    
    # Ears (two circles offset from center)
    ear_offset = 12
    ear_size = 8
    left_ear_x = mx + math.cos(rad + math.pi/4) * ear_offset
    left_ear_y = my + math.sin(rad + math.pi/4) * ear_offset
    right_ear_x = mx + math.cos(rad - math.pi/4) * ear_offset
    right_ear_y = my + math.sin(rad - math.pi/4) * ear_offset
    
    # Ear glow
    pygame.draw.circle(screen, (140, 120, 100), (int(left_ear_x), int(left_ear_y)), ear_size + 2)
    pygame.draw.circle(screen, (140, 120, 100), (int(right_ear_x), int(right_ear_y)), ear_size + 2)
    pygame.draw.circle(screen, MOUSE_EAR_COLOR, (int(left_ear_x), int(left_ear_y)), ear_size)
    pygame.draw.circle(screen, MOUSE_EAR_COLOR, (int(right_ear_x), int(right_ear_y)), ear_size)
    
    # Body (main circle with gradient effect)
    for i in range(3, 0, -1):
        shade = tuple(max(0, c - i * 20) for c in MOUSE_COLOR)
        pygame.draw.circle(screen, shade, (mx, my), MOUSE_SIZE - i * 2)
    pygame.draw.circle(screen, MOUSE_COLOR, (mx, my), MOUSE_SIZE)
    
    # Sci-fi glow around mouse
    glow_surface = pygame.Surface((MOUSE_SIZE * 3, MOUSE_SIZE * 3), pygame.SRCALPHA)
    pygame.draw.circle(glow_surface, (*ACCENT_COLOR, 15), (MOUSE_SIZE * 1.5, MOUSE_SIZE * 1.5), MOUSE_SIZE * 1.5)
    screen.blit(glow_surface, (mx - MOUSE_SIZE * 1.5, my - MOUSE_SIZE * 1.5))
    
    # Nose (pink dot at front)
    nose_x = mx + math.cos(rad) * (MOUSE_SIZE + 3)
    nose_y = my + math.sin(rad) * (MOUSE_SIZE + 3)
    pygame.draw.circle(screen, MOUSE_NOSE_COLOR, (int(nose_x), int(nose_y)), 4)
    pygame.draw.circle(screen, (255, 150, 150), (int(nose_x), int(nose_y)), 2)
    
    # Eyes (two small dots)
    eye_offset = 8
    eye_size = 2
    left_eye_x = mx + math.cos(rad + math.pi/6) * eye_offset
    left_eye_y = my + math.sin(rad + math.pi/6) * eye_offset
    right_eye_x = mx + math.cos(rad - math.pi/6) * eye_offset
    right_eye_y = my + math.sin(rad - math.pi/6) * eye_offset
    pygame.draw.circle(screen, (50, 50, 50), (int(left_eye_x), int(left_eye_y)), eye_size)
    pygame.draw.circle(screen, (50, 50, 50), (int(right_eye_x), int(right_eye_y)), eye_size)


def draw_food(screen, food_x, food_y):
    """Draw cheese wedge with holes"""
    cx = int(ARENA_X + food_x)
    cy = int(ARENA_Y + food_y)
    
    # Cheese wedge (triangle)
    points = [
        (cx - FOOD_SIZE, cy + FOOD_SIZE // 2),
        (cx + FOOD_SIZE, cy + FOOD_SIZE // 2),
        (cx, cy - FOOD_SIZE)
    ]
    
    # Gradient layers for 3D effect
    for i in range(3):
        offset_points = [(p[0], p[1] + i * 2) for p in points]
        shade = tuple(max(0, c - i * 30) for c in CHEESE_COLOR)
        pygame.draw.polygon(screen, shade, offset_points)
    
    # Main cheese color
    pygame.draw.polygon(screen, CHEESE_COLOR, points)
    
    # Cheese holes (circles)
    hole_positions = [
        (cx - 5, cy + 2),
        (cx + 4, cy + 5),
        (cx - 2, cy - 5)
    ]
    for hx, hy in hole_positions:
        pygame.draw.circle(screen, CHEESE_HOLE_COLOR, (hx, hy), 4)
        pygame.draw.circle(screen, (200, 170, 50), (hx, hy), 2)
    
    # Sci-fi glow effect around cheese
    glow_surface = pygame.Surface((FOOD_SIZE * 5, FOOD_SIZE * 5), pygame.SRCALPHA)
    pygame.draw.circle(glow_surface, (255, 220, 100, 25), 
                      (FOOD_SIZE * 2.5, FOOD_SIZE * 2.5), FOOD_SIZE * 2)
    screen.blit(glow_surface, (cx - FOOD_SIZE * 2.5, cy - FOOD_SIZE * 2.5))
    
    # Sparkle effect (animated)
    sparkle = int((time.time() * 2) % 3)
    if sparkle == 0:
        sparkle_pos = [(cx + 10, cy - 8), (cx - 8, cy + 6)]
        for sx, sy in sparkle_pos:
            pygame.draw.circle(screen, (255, 255, 200), (sx, sy), 2)
    
    # Outline for definition
    pygame.draw.polygon(screen, (200, 180, 70), points, 2)


def draw_ring_network(screen, ring_network, font):
    """Visualize the ring attractor network"""
    # Title
    title = font.render("Ring Attractor Network", True, (150, 200, 255))
    screen.blit(title, (RING_CENTER_X - 100, RING_CENTER_Y - RING_RADIUS - 50))
    subtitle = font.render("(Food direction relative to mouse)", True, (120, 120, 140))
    screen.blit(subtitle, (RING_CENTER_X - 115, RING_CENTER_Y - RING_RADIUS - 25))
    
    # Draw ring structure
    for i in range(ring_network.num_neurons):
        angle = (i / ring_network.num_neurons) * 2 * math.pi
        x = RING_CENTER_X + math.cos(angle) * RING_RADIUS
        y = RING_CENTER_Y + math.sin(angle) * RING_RADIUS
        
        # Neuron activity determines color intensity
        activity = ring_network.activity[i]
        intensity = int(abs(activity) * 255)
        # Make colors brighter and more visible
        color = (intensity, intensity // 2 + 80, 50 + intensity // 3)
        
        # Draw neuron with larger base size
        radius = 6 + int(abs(activity) * 12)
        pygame.draw.circle(screen, color, (int(x), int(y)), radius)
        
        # Add subtle glow effect only for highly active neurons
        if abs(activity) > 0.5:
            glow_surface = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*color, 30), 
                             (int(radius * 1.5), int(radius * 1.5)), int(radius * 1.5))
            screen.blit(glow_surface, (int(x) - radius * 1.5, int(y) - radius * 1.5))
        
        # Draw direction labels (N, E, S, W)
        if i == 0:  # 0° (East)
            label = font.render("E", True, TEXT_COLOR)
            screen.blit(label, (int(x) + 15, int(y) - 10))
        elif i == ring_network.num_neurons // 4:  # 90° (South)
            label = font.render("S", True, TEXT_COLOR)
            screen.blit(label, (int(x) - 10, int(y) + 15))
        elif i == ring_network.num_neurons // 2:  # 180° (West)
            label = font.render("W", True, TEXT_COLOR)
            screen.blit(label, (int(x) - 25, int(y) - 10))
        elif i == 3 * ring_network.num_neurons // 4:  # 270° (North)
            label = font.render("N", True, TEXT_COLOR)
            screen.blit(label, (int(x) - 10, int(y) - 25))
    
    # Draw center circle
    pygame.draw.circle(screen, RING_COLOR, (RING_CENTER_X, RING_CENTER_Y), 
                      RING_RADIUS - 30, 2)
    
    # Draw indicator for "straight ahead" (0° relative = mouse's forward direction)
    indicator_text = font.render("Mouse Forward →", True, (100, 200, 255))
    screen.blit(indicator_text, (RING_CENTER_X + RING_RADIUS + 20, RING_CENTER_Y - 10))
    # Draw arrow pointing to 0° position (East on the ring)
    pygame.draw.polygon(screen, (100, 200, 255), [
        (RING_CENTER_X + RING_RADIUS + 15, RING_CENTER_Y),
        (RING_CENTER_X + RING_RADIUS + 5, RING_CENTER_Y - 5),
        (RING_CENTER_X + RING_RADIUS + 5, RING_CENTER_Y + 5)
    ])


def draw_motor_network(screen, motor_network, font):
    """Visualize the motor output network - elegant and compact"""
    # Title - positioned above neurons
    title = font.render("Motor Output Layer", True, (100, 255, 180))
    screen.blit(title, (MOTOR_VIS_X - 90, MOTOR_VIS_Y - 50))
    subtitle = font.render("(Learns via Reward-Modulated Hebbian)", True, (100, 120, 140))
    screen.blit(subtitle, (MOTOR_VIS_X - 130, MOTOR_VIS_Y - 25))
    
    motor_labels = ["Turn Left", "Straight", "Turn Right"]
    
    # More spacing between neurons
    neuron_spacing = 180
    
    for i in range(MOTOR_NEURONS):
        x = MOTOR_VIS_X - neuron_spacing + i * neuron_spacing
        y = MOTOR_VIS_Y
        
        # Activity determines size and brightness (clamped to reasonable range)
        activity = motor_network.activity[i]
        normalized_activity = (activity - motor_network.activity.min()) / (motor_network.activity.max() - motor_network.activity.min() + 1e-6)
        
        # Smaller, more controlled size range
        base_size = 18
        max_size_increase = 12
        size = base_size + int(normalized_activity * max_size_increase)
        
        brightness = int(normalized_activity * 150) + 80
        color = tuple(min(255, int(c * brightness / 255)) for c in MOTOR_COLORS[i])
        
        # Elegant glow effect (much more subtle)
        if normalized_activity > 0.4:
            glow_surface = pygame.Surface((size * 3, size * 3), pygame.SRCALPHA)
            glow_intensity = int(normalized_activity * 25)
            pygame.draw.circle(glow_surface, (*color, glow_intensity), 
                             (size * 1.5, size * 1.5), size * 1.5)
            screen.blit(glow_surface, (x - size * 1.5, y - size * 1.5))
        
        # Draw motor neuron (smaller, more elegant)
        pygame.draw.circle(screen, color, (x, y), size)
        pygame.draw.circle(screen, tuple(min(255, c + 50) for c in color), (x, y), size, 2)
        
        # Label below neuron (even more space to avoid overlap)
        label_font = pygame.font.Font(None, 22)
        label = label_font.render(motor_labels[i], True, TEXT_COLOR)
        label_rect = label.get_rect(center=(x, y + size + 40))
        screen.blit(label, label_rect)
        
        # Activity value above neuron (more space)
        value_font = pygame.font.Font(None, 20)
        value_text = value_font.render(f"{activity:.2f}", True, color)
        value_rect = value_text.get_rect(center=(x, y - size - 20))
        screen.blit(value_text, value_rect)


def draw_arena(screen, font):
    """Draw the game arena with sci-fi grid"""
    # Arena label with glow
    label_font = pygame.font.Font(None, 28)
    arena_label = label_font.render("NAVIGATION ARENA", True, (0, 255, 200))
    screen.blit(arena_label, (ARENA_X + 10, ARENA_Y - 35))
    
    # Arena background with gradient
    pygame.draw.rect(screen, ARENA_COLOR, 
                    (ARENA_X, ARENA_Y, ARENA_SIZE, ARENA_SIZE))
    
    # Sci-fi grid lines with fading effect
    grid_spacing = 50
    for i in range(0, ARENA_SIZE, grid_spacing):
        # Vary line brightness for depth effect
        brightness = 30 + (i % 100)
        grid_color = (brightness, brightness + 10, brightness + 30)
        pygame.draw.line(screen, grid_color,
                        (ARENA_X + i, ARENA_Y),
                        (ARENA_X + i, ARENA_Y + ARENA_SIZE), 1)
        pygame.draw.line(screen, grid_color,
                        (ARENA_X, ARENA_Y + i),
                        (ARENA_X + ARENA_SIZE, ARENA_Y + i), 1)
    
    # Glowing border
    pygame.draw.rect(screen, (80, 120, 180),
                    (ARENA_X, ARENA_Y, ARENA_SIZE, ARENA_SIZE), 3)
    pygame.draw.rect(screen, ACCENT_COLOR,
                    (ARENA_X - 1, ARENA_Y - 1, ARENA_SIZE + 2, ARENA_SIZE + 2), 1)


def draw_mini_plot(screen, data, x, y, width, height, title, color, font, ylabel=""):
    """Draw a compact line plot for learning metrics with sci-fi styling"""
    if len(data) < 2:
        # Show placeholder with animated loading
        pygame.draw.rect(screen, (15, 20, 30), (x, y, width, height))
        pygame.draw.rect(screen, ACCENT_COLOR, (x, y, width, height), 2)
        msg = font.render("INITIALIZING...", True, ACCENT_COLOR)
        screen.blit(msg, (x + width // 2 - 50, y + height // 2 - 10))
        return
    
    # Background with gradient effect
    pygame.draw.rect(screen, (15, 20, 30), (x, y, width, height))
    pygame.draw.rect(screen, ACCENT_COLOR, (x, y, width, height), 2)
    pygame.draw.rect(screen, (50, 70, 100), (x, y, width, height), 1)
    
    # Title
    title_text = font.render(title, True, TEXT_COLOR)
    screen.blit(title_text, (x + 10, y + 8))
    
    # Find min/max for scaling
    min_val = min(data)
    max_val = max(data)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # Y-axis labels
    max_label = font.render(f"{max_val:.0f}", True, (120, 120, 140))
    screen.blit(max_label, (x + 8, y + 28))
    min_label = font.render(f"{min_val:.0f}", True, (120, 120, 140))
    screen.blit(min_label, (x + 8, y + height - 20))
    
    # Plot area
    plot_x = x + 45
    plot_y = y + 30
    plot_width = width - 55
    plot_height = height - 40
    
    # Plot data points
    points = []
    for i, value in enumerate(data):
        px = plot_x + (i / (len(data) - 1)) * plot_width
        normalized = (value - min_val) / val_range
        py = plot_y + plot_height - normalized * plot_height
        points.append((int(px), int(py)))
    
    # Draw line
    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 2)
    
    # Draw points
    for point in points[-10:]:  # Highlight recent points
        pygame.draw.circle(screen, color, point, 3)
    
    # Current value indicator
    if data:
        current_text = font.render(f"Current: {data[-1]:.0f}", True, color)
        screen.blit(current_text, (x + width - 100, y + 8))


def draw_learning_plots(screen, tracker, font):
    """Draw all learning visualization plots"""
    # Section header with sci-fi styling
    header_font = pygame.font.Font(None, 30)
    header = header_font.render("◢ LEARNING PROGRESS METRICS ◣", True, ACCENT_COLOR)
    screen.blit(header, (PLOT_X + 250, PLOT_Y - 45))
    
    # Wider plots since we only have 2 now
    plot_width = 500
    
    # Plot 1: Steps per Episode (lower is better - shows learning efficiency)
    draw_mini_plot(screen, tracker.episode_steps,
                  PLOT_X, PLOT_Y, plot_width, PLOT_HEIGHT,
                  "Steps to Reach Food", (100, 180, 255), font)
    
    # Plot 2: Moving Average (smoothed learning curve)
    draw_mini_plot(screen, tracker.avg_steps_history,
                  PLOT_X + plot_width + 30, PLOT_Y, plot_width, PLOT_HEIGHT,
                  "Learning Efficiency (5-ep avg)", (100, 255, 180), font)


def draw_stats(screen, mouse, episode_steps, tracker, font):
    """Draw statistics panel"""
    stats_x = 800
    stats_y = 790
    
    stats = [
        f"Episode: {mouse.episode} | Steps: {episode_steps} | Total Reward: {mouse.total_reward:.1f}",
        f"Exploration: {mouse.exploration_rate * 100:.1f}% random actions (adaptive) | Mouse: ({int(mouse.x)}, {int(mouse.y)}) @ {int(mouse.angle)}°"
    ]
    
    if tracker.episode_steps:
        avg = tracker.get_recent_avg_steps()
        stats.append(f"Recent Avg Steps: {avg:.1f} (learning efficiency improving = fewer steps needed)")
    
    for i, stat in enumerate(stats):
        text = font.render(stat, True, TEXT_COLOR)
        screen.blit(text, (stats_x, stats_y + i * 22))


# ==================== MAIN GAME LOOP ====================
def main():
    print(colored("Starting CAN Navigation Game...", "cyan", attrs=["bold"]))
    print(colored("Mouse will learn to navigate to food using a ring attractor network!", "green"))
    
    pygame.init()
    
    # Fullscreen window
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CAN Navigation - Biologically Inspired Learning")
    
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    title_font = pygame.font.Font(None, 48)
    
    # Initialize mouse and food
    mouse = Mouse(MOUSE_START_X, MOUSE_START_Y)
    food_x = np.random.randint(100, ARENA_SIZE - 100)
    food_y = np.random.randint(100, ARENA_SIZE - 100)
    
    # Create speed slider
    speed_slider = Slider(1550, 100, 300, 1, 20, 1, "Speed")
    
    # Create learning tracker
    tracker = LearningTracker(max_history=50)
    
    episode_steps = 0
    running = True
    
    print(colored("\nGame started! Watch the mouse learn to navigate!", "yellow"))
    print(colored("Close the window to exit.", "yellow"))
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:  # Reset
                    mouse.reset_position(episode_steps)
                    food_x = np.random.randint(100, ARENA_SIZE - 100)
                    food_y = np.random.randint(100, ARENA_SIZE - 100)
                    episode_steps = 0
            
            # Handle slider events
            speed_slider.handle_event(event)
        
        # Clear screen
        screen.fill(BG_COLOR)
        
        # Title (centered) with sci-fi glow
        title = title_font.render("CONTINUOUS ATTRACTOR NETWORK  •  NAVIGATION", True, ACCENT_COLOR)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 50))
        # Glow effect for title
        for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
            glow_title = title_font.render("CONTINUOUS ATTRACTOR NETWORK  •  NAVIGATION", True, (0, 100, 120))
            glow_rect = glow_title.get_rect(center=(SCREEN_WIDTH // 2 + offset[0], 50 + offset[1]))
            screen.blit(glow_title, glow_rect)
        screen.blit(title, title_rect)
        
        # Subtitle with elegant sci-fi styling
        subtitle_font = pygame.font.Font(None, 22)
        subtitle = subtitle_font.render("Biologically-Inspired Learning  ▸  Ring Attractors  ▸  Reward-Modulated Hebbian Plasticity", True, (120, 180, 220))
        subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 90))
        screen.blit(subtitle, subtitle_rect)
        
        # Run simulation based on speed slider value
        steps_this_frame = int(speed_slider.value)
        for _ in range(steps_this_frame):
            # Mouse thinks and acts
            distance_to_food = mouse.think_and_act(food_x, food_y)
            episode_steps += 1
            
            # Check if food is reached
            if distance_to_food < FOOD_REWARD_DISTANCE:
                # Big reward!
                mouse.motor_network.learn(1.0)
                mouse.total_reward += 1.0
                
                # Log episode metrics
                tracker.log_episode(episode_steps, 1.0)
                
                print(colored(f"Episode {mouse.episode} complete! Steps: {episode_steps}, Avg: {tracker.get_recent_avg_steps():.1f}", "green", attrs=["bold"]))
                
                # Reset for new episode
                mouse.reset_position(episode_steps)
                food_x = np.random.randint(100, ARENA_SIZE - 100)
                food_y = np.random.randint(100, ARENA_SIZE - 100)
                episode_steps = 0
                break  # Exit speed loop if food is reached
        
        # Draw everything
        draw_arena(screen, font)
        draw_food(screen, food_x, food_y)
        draw_mouse(screen, mouse)
        draw_ring_network(screen, mouse.ring_network, font)
        draw_motor_network(screen, mouse.motor_network, font)
        draw_learning_plots(screen, tracker, font)
        draw_stats(screen, mouse, episode_steps, tracker, font)
        speed_slider.draw(screen, font)
        
        # Instructions (centered at bottom)
        instructions = font.render("Press R to reset | ESC to quit | Drag slider to change speed", True, (150, 150, 170))
        instructions_rect = instructions.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 25))
        screen.blit(instructions, instructions_rect)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    print(colored("\nGame ended. Thanks for playing!", "cyan", attrs=["bold"]))


if __name__ == "__main__":
    main()

