import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class Buffalo:
    def __init__(self, x_position, y_position, vote_angle):
        self.position = (x_position, y_position)
        self.vote = vote_angle
        self.speed = 2 

    
    def update(self):
        # Update Buffalo's position
        x = np.cos(self.vote)
        y = np.sin(self.vote)
        self.position +=  np.array([x, y])

def generate_herd(plane_length, herd_positions_set):
    range_herd = plane_length
    while True:
        # Generate unique position (x, y)
        x = np.random.uniform(0, range_herd)
        y = np.random.uniform(0, range_herd)
        
        new_position = (x, y)
        
        # Ensure position uniqueness
        if new_position not in herd_positions_set:
            vote = np.random.uniform(0, 360)
            herd_positions_set.add(new_position)  # Track position
            return Buffalo(x, y, vote)

def plot_herd(herd, plane_length):
    # Extract positions and vote angles from the herd
    x_positions = [Buffalo.position[0] for Buffalo in herd]
    y_positions = [Buffalo.position[1] for Buffalo in herd]
    angles = [Buffalo.vote for Buffalo in herd]

    # Convert vote angles to radians for trigonometric calculations
    angle_radians = np.radians(angles)
    
    # Set vector length for each buffalo
    vector_length = plane_length / 50
    
    # Calculate vector components (dx, dy) based on angles
    dx = np.cos(angle_radians)
    dy = np.sin(angle_radians)

    # Normalize vectors and scale them to desired length
    magnitude = np.sqrt(dx**2 + dy**2)
    dx_scaled = (dx / magnitude) * vector_length
    dy_scaled = (dy / magnitude) * vector_length

    # Calculate the average direction of the herd's votes
    center_coor = plane_length / 2
    avg_vote = np.mean(angles)
    avg_angle_radians = np.radians(avg_vote)

    # Average direction of the herd's vote
    dx_avg = np.mean(dx_scaled)
    dy_avg = np.mean(dy_scaled)

    # Normalize the average direction and scale to 1/3 of plane length
    avg_magnitude = np.sqrt(dx_avg**2 + dy_avg**2)
    dx_avg_scaled = (dx_avg / avg_magnitude) * (plane_length / 2)
    dy_avg_scaled = (dy_avg / avg_magnitude) * (plane_length / 2)
    
    # Plot the buffalo vectors
    plt.figure(figsize=(8, 8))
    plt.quiver(x_positions, y_positions, dx_scaled, dy_scaled, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6)
    
    # Plot the average vector (red)
    plt.quiver(center_coor, center_coor, dx_avg_scaled, dy_avg_scaled, angles='xy', scale_units='xy', scale=1, color='red', alpha=0.6) 
    
    plt.title("Buffalo Herd Vectors")
    plt.xlabel("Plane X (m)")
    plt.ylabel("Plane Y (m)")
    plt.grid(True)
    plt.show()

# Parameters
herd_size = 200
plane_length = 2000

# Initialize herd and position set
herd = []
herd_positions_set = set()


# Generate herd with unique positions
for _ in range(herd_size):
    herd.append(generate_herd(plane_length, herd_positions_set))


votess = []
for i in range(herd_size):
    votess.append(herd[i].vote)

# Plot the herd and average vector
plot_herd(herd, plane_length)
