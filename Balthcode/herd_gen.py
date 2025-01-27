import numpy as np
from buffalo_gen import Buffalo

def generate_herd(plane_length, herd_positions): 
    while True:
        # Generate unique position (x, y)
        r = np.sqrt(np.random.uniform(0, 1)) * plane_length
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        new_position = [x, y]
        
        # Ensure position uniqueness
        if new_position not in herd_positions:
            herd_positions.append(new_position)  # Track position
            return Buffalo(x, y)