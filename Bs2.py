"""
Enhanced Multi-Drone ATC System with Attention DQN and Parameter Sharing
"""

# ----------
# IMPORTS
# ----------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pygame
from datetime import datetime
import math
import argparse

# ----------
# DRONE CLASS
# ----------
class Drone:
    def __init__(self, drone_id, start_pos, goal_pos):
        self.id = drone_id
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.reset()
        
    def reset(self):
        self.pos = self.start_pos
        self.collided = False
        self.at_goal = False
        self.active = True
        self.steps_since_progress = 0
        self.travel_distance = 0

# ----------
# ENVIRONMENT CLASS
# ----------
class MultiDroneEnv:
    def __init__(self, grid_size=20, num_drones=4):
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.start_positions = [(0,0), (0,19), (19,0), (19,19)]
        self.goal_positions = [(7,7), (13,13), (7,13), (13,7)]
        self.drones = [Drone(i, self.start_positions[i], self.goal_positions[i]) 
                      for i in range(num_drones)]
        self.obstacles = []
        self.episode_count = 0

    def reset(self):
        self.episode_count += 1
        self._generate_obstacles()
        for drone in self.drones:
            drone.reset()
        return self.get_states()

    def _generate_obstacles(self):
        self.obstacles = []
        existing_pos = set(self.start_positions + self.goal_positions)
        
        # Curriculum learning
        base_obstacles = 5
        max_obstacles = 20
        num_obstacles = min(max_obstacles, base_obstacles + self.episode_count//100)
        
        while len(self.obstacles) < num_obstacles:
            x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            if (x,y) not in existing_pos and (x,y) not in self.obstacles:
                self.obstacles.append((x,y))

    def get_states(self):
        return [self._get_drone_state(drone) for drone in self.drones]

    def _get_drone_state(self, drone):
        state = [
            drone.pos[0]/self.grid_size,
            drone.pos[1]/self.grid_size,
            drone.goal_pos[0]/self.grid_size,
            drone.goal_pos[1]/self.grid_size,
            drone.travel_distance/(self.grid_size*2)
        ]
        
        # Enhanced proximity sensing
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0: continue
                nx = drone.pos[0] + dx
                ny = drone.pos[1] + dy
                obstacle = 1 if (nx, ny) in self.obstacles else 0
                other_drone = sum(1 for other in self.drones 
                                if other != drone and abs(other.pos[0]-nx) <= 2 
                                and abs(other.pos[1]-ny) <= 2)
                state.append(float(obstacle or other_drone))
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        old_positions = [drone.pos for drone in self.drones]
        new_positions = []
        collisions = []
        
        # Process movements
        for i, action in enumerate(actions):
            drone = self.drones[i]
            if not drone.active:
                new_positions.append(drone.pos)
                continue
                
            new_pos = self._move(drone.pos, action)
            if not self._is_valid_position(new_pos):
                new_pos = drone.pos
                collisions.append(i)
            new_positions.append(new_pos)

        # Collision detection
        position_count = {}
        for pos in new_positions:
            position_count[pos] = position_count.get(pos, 0) + 1
            
        final_positions = []
        for i, pos in enumerate(new_positions):
            if position_count[pos] > 1 or i in collisions:
                final_positions.append(self.drones[i].pos)
                self.drones[i].collided = True
            else:
                final_positions.append(pos)

        # Update positions and rewards
        rewards = np.zeros(self.num_drones)
        for i, drone in enumerate(self.drones):
            old_pos = old_positions[i]
            new_pos = final_positions[i]
            drone.pos = new_pos
            drone.travel_distance += math.hypot(new_pos[0]-old_pos[0], new_pos[1]-old_pos[1])
            
            if new_pos == drone.goal_pos:
                rewards[i] += 2000
                drone.at_goal = True
            elif drone.collided:
                rewards[i] -= 50
            else:
                old_dist = math.hypot(old_pos[0]-drone.goal_pos[0], old_pos[1]-drone.goal_pos[1])
                new_dist = math.hypot(new_pos[0]-drone.goal_pos[0], new_pos[1]-drone.goal_pos[1])
                
                rewards[i] += (old_dist - new_dist) * 2.0
                rewards[i] -= 0.1
                
                if new_dist < 5:
                    rewards[i] += (5 - new_dist) * 0.5
                
                if new_dist >= old_dist:
                    drone.steps_since_progress += 1
                    if drone.steps_since_progress > 15:
                        rewards[i] -= 1.5
                else:
                    drone.steps_since_progress = 0

            if not self._is_valid_position(new_pos):
                rewards[i] -= 10

        done = all(not drone.active for drone in self.drones)
        for drone in self.drones:
            drone.active = not (drone.collided or drone.at_goal)
            
        return self.get_states(), rewards, done, {}

    def _move(self, pos, action):
        x, y = pos
        if action == 0: return (x, y+1)  # Up
        if action == 1: return (x, y-1)  # Down
        if action == 2: return (x-1, y)  # Left
        if action == 3: return (x+1, y)  # Right
        return (x, y)

    def _is_valid_position(self, pos):
        return (0 <= pos[0] < self.grid_size and 
                0 <= pos[1] < self.grid_size and 
                pos not in self.obstacles)

# ----------
# ATTENTION DQN ARCHITECTURE
# ----------
class AttentionDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        attn_output, _ = self.attention(features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0))
        return self.decoder(attn_output.squeeze(0))

# ----------
# SHARED PARAMETER AGENT
# ----------
class SharedDQNAgent:
    def __init__(self, input_size, output_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionDQN(input_size, output_size).to(self.device)
        self.target_model = AttentionDQN(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.memory = deque(maxlen=100000)
        self.batch_size = 512
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.update_freq = 50

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ----------
# VISUALIZATION SYSTEM
# ----------
class DroneSimulation:
    def __init__(self, grid_size=20, cell_size=30):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.colors = {
            'start': (255, 100, 100),
            'goal': (100, 255, 100),
            'drone': (100, 100, 255),
            'obstacle': (50, 50, 50),
            'background': (245, 245, 245),
            'grid': (220, 220, 220)
        }
        pygame.init()
        self.screen = pygame.display.set_mode(
            (grid_size*cell_size, grid_size*cell_size))
        pygame.display.set_caption("Drone ATC Simulation")
        self.font = pygame.font.Font(None, 24)

    def draw_env(self, env):
        self.screen.fill(self.colors['background'])
        
        # Draw grid lines
        for x in range(env.grid_size + 1):
            pygame.draw.line(self.screen, self.colors['grid'],
                            (x * self.cell_size, 0),
                            (x * self.cell_size, env.grid_size * self.cell_size))
        for y in range(env.grid_size + 1):
            pygame.draw.line(self.screen, self.colors['grid'],
                            (0, y * self.cell_size),
                            (env.grid_size * self.cell_size, y * self.cell_size))

        # Draw obstacles
        for (x, y) in env.obstacles:
            pygame.draw.rect(self.screen, self.colors['obstacle'],
                            pygame.Rect(
                                x * self.cell_size + 1,
                                y * self.cell_size + 1,
                                self.cell_size - 2,
                                self.cell_size - 2
                            ))

        # Draw start and goal positions
        for i, drone in enumerate(env.drones):
            # Start position
            sx, sy = drone.start_pos
            pygame.draw.rect(self.screen, self.colors['start'],
                           pygame.Rect(
                               sx * self.cell_size + 2,
                               sy * self.cell_size + 2,
                               self.cell_size - 4,
                               self.cell_size - 4
                           ), 2)
            
            # Goal position
            gx, gy = drone.goal_pos
            pygame.draw.circle(self.screen, self.colors['goal'],
                             (int((gx+0.5)*self.cell_size), 
                              int((gy+0.5)*self.cell_size)), 8)
            text = self.font.render(str(i), True, (255,255,255))
            self.screen.blit(text, 
                            (int((gx+0.5)*self.cell_size)-5, 
                             int((gy+0.5)*self.cell_size)-8))

        # Draw drones
        for i, drone in enumerate(env.drones):
            x, y = drone.pos
            color = self.colors['drone'] if not drone.collided else (80, 80, 80)
            pygame.draw.circle(self.screen, color,
                             (int((x+0.5)*self.cell_size), 
                              int((y+0.5)*self.cell_size)), self.cell_size//3)
            text = self.font.render(str(i), True, (255,255,255))
            self.screen.blit(text, 
                            (int((x+0.5)*self.cell_size)-5, 
                             int((y+0.5)*self.cell_size)-8))
        
        # Info panel
        text = self.font.render(f"Active: {sum(d.active for d in env.drones)}", True, (0,0,0))
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()

# ----------
# TRAINING LOGGER
# ----------
class TrainingLogger:
    def __init__(self):
        self.filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.filename, 'w') as f:
            f.write("episode,total_reward,steps,collisions,success_rate,timeout,avg_distance,epsilon\n")
            
    def log(self, episode, total_reward, steps, collisions, success_rate, timeout, avg_distance, epsilon):
        with open(self.filename, 'a') as f:
            f.write(f"{episode},{total_reward:.1f},{steps},{collisions},{success_rate:.2f},"
                   f"{int(timeout)},{avg_distance:.2f},{epsilon:.3f}\n")

# ----------
# TRAINING LOOP
# ----------
def train(render=True):
    env = MultiDroneEnv(grid_size=20)
    sim = None
    if render:
        sim = DroneSimulation(grid_size=20, cell_size=30)
    
    logger = TrainingLogger()
    state_size = len(env._get_drone_state(env.drones[0]))
    action_size = 5
    agent = SharedDQNAgent(state_size, action_size)
    
    episodes = 100  
    
    max_steps = 500
    update_target_every = 25
    running = True  # Control flag for visualization
    
    for episode in range(episodes):
        if not running:
            break
            
        states = env.reset()
        total_rewards = np.zeros(env.num_drones)
        step_count = 0
        collisions = 0
        done = False
        
        while not done and step_count < max_steps and running:
            # Action selection
            actions = []
            for i, drone in enumerate(env.drones):
                if not drone.active:
                    actions.append(4)
                else:
                    actions.append(agent.act(states[i]))
            
            # Environment step
            next_states, rewards, done, _ = env.step(actions)
            
            # Store experiences
            for i in range(env.num_drones):
                if env.drones[i].active:
                    agent.remember(states[i], actions[i], 
                                 rewards[i], next_states[i], done)
            
            # Train agent
            agent.replay()
            
            # Update metrics
            total_rewards += rewards
            step_count += 1
            collisions += sum(1 for drone in env.drones if drone.collided)
            
            # Visualization
            if render and sim:
                try:
                    sim.draw_env(env)
                    pygame.time.wait(50)
                    # Event handling
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            pygame.quit()
                            sim = None
                            break
                except pygame.error:
                    running = False
                    render = False
                    sim = None
            
            states = next_states

        # Post-episode updates
        agent.epsilon = max(agent.epsilon_min, 
                           agent.epsilon * agent.epsilon_decay)
        
        if episode % update_target_every == 0:
            agent.update_target()
        
        # Calculate metrics
        success_count = sum(1 for drone in env.drones if drone.at_goal)
        success_rate = success_count / env.num_drones
        timeout = step_count >= max_steps
        avg_distance = sum(math.hypot(d.pos[0]-d.goal_pos[0], d.pos[1]-d.goal_pos[1]) 
                      for d in env.drones) / env.num_drones
        
        # Logging
        logger.log(episode+1, sum(total_rewards), step_count, collisions,
                  success_rate, timeout, avg_distance, agent.epsilon)
        
        print(f"Ep {episode+1:04d} | R:{sum(total_rewards):07.1f} | "
              f"Steps:{step_count:03d} | Coll:{collisions:02d} | "
              f"Succ:{success_rate:.2f} | Dist:{avg_distance:.1f} | "
              f"ε:{agent.epsilon:.3f}")

    # Cleanup
    if sim is not None:
        pygame.quit()

# ----------
# MAIN EXECUTION
# ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Enable visualization during training')
    args = parser.parse_args()
    
    # Phase 1: Training without visualization for speed
    print("Starting training phase...")
    train(render=args.render)
    
    # Phase 2: Final evaluation with visualization
    if not args.render:
        print("\n=== Starting final evaluation with visualization ===")
        train(render=True)