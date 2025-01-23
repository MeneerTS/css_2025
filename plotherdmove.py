
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class Buffalo:
    def __init__(self, x_position, y_position, vote_angle):
        self.position = np.array([x_position, y_position])  # Use numpy array for easier manipulation
        self.vote = vote_angle
        self.speed = 1  # Set speed to control how fast the buffalo moves
    
    def update(self):
        # Update Buffalo's position based on its vote (direction)
        x = np.cos(np.radians(self.vote)) * self.speed
        y = np.sin(np.radians(self.vote)) * self.speed
        self.position += np.array([x, y])  # Update position by moving in the vote direction

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

def plot_herd_as_vectors(herd, plane_length):
    # Extract positions and vote angles from the herd
    x_positions = [buffalo.position[0] for buffalo in herd]
    y_positions = [buffalo.position[1] for buffalo in herd]
    angles = [buffalo.vote for buffalo in herd]

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

    return x_positions, y_positions, dx_scaled, dy_scaled

# Parameters
herd_size = 350
plane_length = 2000

# Initialize herd and position set
herd = []
herd_positions_set = set()

# Generate herd with unique positions
for _ in range(herd_size):
    herd.append(generate_herd(plane_length, herd_positions_set))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, plane_length)
ax.set_ylim(0, plane_length)

# Initialize the plot
x_positions, y_positions, dx_scaled, dy_scaled = plot_herd_as_vectors(herd, plane_length)
quiver = ax.quiver(x_positions, y_positions, dx_scaled, dy_scaled, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6)

# Define the update function for the animation
def update(frame):
    # Update positions of the buffaloes
    for buffalo in herd:
        buffalo.update()

    # Get new positions and vectors for the buffaloes
    x_positions, y_positions, dx_scaled, dy_scaled = plot_herd_as_vectors(herd, plane_length)
    
    # Update the quiver plot with new data
    quiver.set_offsets(np.column_stack((x_positions, y_positions)))
    quiver.set_UVC(dx_scaled, dy_scaled)
    return quiver,

# Create the animation
ani = FuncAnimation(fig, update, frames=2000, interval=12, blit=True)

# Convert the animation to HTML5 video and display it
HTML(ani.to_html5_video())