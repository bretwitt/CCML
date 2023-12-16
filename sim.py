import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
number_of_agents = 150
simulation_duration = 5  # in seconds
time_step = 1./240.  # simulation time step
# Wall Parameters
wall_thickness = 0.2
wall_height = 2
arena_size = 10  # Length and width of the arena

friction_coefficient = 0.01

# Agent Class
class Agent:
    def __init__(self, agent_id, state, velocity=None):
        self.agent_id = agent_id
        self.state = state
        self.velocity = velocity

# Initialize PyBullet
p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")


### CREATE WALLS ####

wall_shapes = []
wall_bodies = []

# Bottom Wall
bottom_wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2])
bottom_wall_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bottom_wall_shape, basePosition=[0, -arena_size / 2, wall_height / 2])
wall_shapes.append(bottom_wall_shape)
wall_bodies.append(bottom_wall_body)

# Top Wall
top_wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2])
top_wall_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=top_wall_shape, basePosition=[0, arena_size / 2, wall_height / 2])
wall_shapes.append(top_wall_shape)
wall_bodies.append(top_wall_body)

# Left Wall
left_wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2])
left_wall_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=left_wall_shape, basePosition=[-arena_size / 2, 0, wall_height / 2])
wall_shapes.append(left_wall_shape)
wall_bodies.append(left_wall_body)

# Right Wall
right_wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2])
right_wall_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=right_wall_shape, basePosition=[arena_size / 2, 0, wall_height / 2])
wall_shapes.append(right_wall_shape)
wall_bodies.append(right_wall_body)

#### CREATE AGENTS ####

# Create agents
agents = []
positions = {i: [] for i in range(number_of_agents)}
agent_id_to_index = {}  # Mapping from PyBullet ID to index

def generate_start_position(front_density, back_density, arena_size):
    # Determine the density gradient
    density_gradient = np.linspace(front_density, back_density, arena_size)

    # Randomly select a position along the y-axis based on the density gradient
    x_pos = np.random.choice(arena_size, p=density_gradient/np.sum(density_gradient)) - arena_size / 2 - 0.1

    # Random x position
    y_pos = np.random.uniform(-arena_size / 2, arena_size / 2)

    return [x_pos, y_pos, 0.25]

def calculate_local_density(agent_id, agents, radius):
    count = 0
    pos_i, _ = p.getBasePositionAndOrientation(agent_id)
    for agent in agents:
        if agent.agent_id != agent_id:
            pos_j, _ = p.getBasePositionAndOrientation(agent.agent_id)
            distance = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
            if distance < radius:
                count += 1
    density = count / (np.pi * radius**2)
    return density

for i in range(number_of_agents):
    start_position = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.25]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])

    agent_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=0.5)
    agent_body = p.createMultiBody(baseMass=80,
                                   baseCollisionShapeIndex=agent_shape,
                                   basePosition=start_position,
                                   baseOrientation=start_orientation)

    p.changeDynamics(agent_body, -1, angularDamping=1, linearDamping=0.1)
    #p.setAngularFactor(agent_body, [0, 0, 0])
    
    if random.random() < 0.5:
        state = "WALK"
    else:
        state = "IDLE"

    velocity = [np.random.uniform(0, 15), 0, 0] if state == "WALK" else None
    
    agent = Agent(agent_body, state, velocity)

    agents.append(agent)
    agent_id_to_index[agent_body] = i  # Map PyBullet ID to index
    positions[i].append(start_position[:2])


# Function to update agent velocity
def update_agent_velocity(agent, velocity):
    velocity_with_fixed_z = [velocity[0], velocity[1], 0]  # Freezing the Z axis
    p.resetBaseVelocity(agent.agent_id, linearVelocity=velocity_with_fixed_z)

# Repulsion Parameters
repulsion_radius = 0.25  # Radius within which agents repel each other
repulsion_strength = 5  # Strength of the repulsion

# Function to apply repulsive force between agents
def apply_repulsion():
    for i, agent_i in enumerate(agents):
        pos_i, _ = p.getBasePositionAndOrientation(agent_i.agent_id)
        for j, agent_j in enumerate(agents):
            if i == j:
                continue
            pos_j, _ = p.getBasePositionAndOrientation(agent_j.agent_id)
            distance = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
            if distance < repulsion_radius:
                # Calculate repulsive force direction
                direction = np.array(pos_i) - np.array(pos_j)
                direction /= np.linalg.norm(direction)
                # Calculate force magnitude based on distance
                force_magnitude = repulsion_strength * (1 - distance / repulsion_radius)
                force = direction * force_magnitude
                # Update velocities
                if agent_i.state == "WALK":
                    agent_i.velocity += force
                if agent_j.state == "WALK":
                    agent_j.velocity -= force

def apply_friction(agent):
    if agent.velocity is not None:
        # Calculate the magnitude of the velocity
        velocity_magnitude = np.linalg.norm(agent.velocity)
        if velocity_magnitude > 0:
            # Calculate the friction force
            friction_force = -friction_coefficient * np.array(agent.velocity)
            # Apply the friction force to the agent's velocity
            agent.velocity += friction_force

data = []

# Simulation loop
for step in range(int(simulation_duration / time_step)):
    #apply_repulsion()
    for agent in agents:
        #apply_friction(agent)
        density = calculate_local_density(agent.agent_id, agents, 1.0)

        pos, _ = p.getBasePositionAndOrientation(agent.agent_id)

        vel, ang_vel = p.getBaseVelocity(agent.agent_id)

        data.append([agent.agent_id, pos, vel , agent.state, density])

        # Check if agent is at the edge of the arena
        if abs(pos[0]) > arena_size / 2 - 1.0 or abs(pos[1]) > arena_size / 2 - 1.0:
            agent.state = "IDLE"
            agent.velocity = [0, 0, 0]

        if agent.state == "WALK":
            update_agent_velocity(agent, agent.velocity)
        else:  # IDLE state
            update_agent_velocity(agent, [0, 0, 0])

        # Record position
        index = agent_id_to_index[agent.agent_id]
        positions[index].append(pos[:2])

    p.stepSimulation()
    time.sleep(time_step)
# Disconnect
p.disconnect()

import csv

with open('agent_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['AgentID', 'Position', 'Velocity', 'State', 'Density'])
    for row in data:
        writer.writerow(row)

# Plotting the trajectories
plt.figure(figsize=(10, 10))
for i in range(number_of_agents):
    traj = np.array(positions[i])
    plt.plot(traj[:, 0], traj[:, 1], label=f'Agent {i}')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectories of Agents in Crowd Simulation')
plt.show()

