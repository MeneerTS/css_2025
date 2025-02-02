import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import networkx as nx

class Buffalo:
    def __init__(self, x_position, y_position):
        self.x = x_position
        self.y = y_position
        self.position = [x_position, y_position]
        self.decision = None  # Initialize decision to None
        
        a = np.random.uniform(0, 1)
        if a > .45:
            self.vote = 0  # not an active voter
        else:
            b = np.random.choice([0, 1])
            if b == 1:
                self.vote = 1  # votes for A
            else:
                self.vote = 2  # votes for B

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

def simulate_herd(r, herd_size, plane_length=1000):
    # Initialize herd and position set
    herd = []
    herd_positions = []
    
    # Generate a herd
    for i in range(herd_size):
        herd.append(generate_herd(plane_length, herd_positions))
    
    # Plot the herd
    color_map = {0: 'grey', 1: 'blue', 2: 'red'}
    voting_record = {0: 'no vote', 1: 'vote A', 2: 'vote B'}
    
    plt.figure(figsize=(8, 7))
    for buff in herd:
        plt.scatter(buff.x, buff.y, s=1.2, color=color_map[buff.vote], label=voting_record[buff.vote])
    
    # Adding legend and labels
    plt.xlabel('Plane (X-axis)')
    plt.ylabel('Plane (Y-axis)')
    plt.xlim(-1.2 * plane_length, 1.2 * plane_length)
    plt.ylim(-1.2 * plane_length, 1.2 * plane_length)
    plt.title('Scatter plot with color-coded points')
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label=f'{voting_record[v]}', 
                   markerfacecolor=color, markersize=10)
        for v, color in color_map.items()],
               title='Legend', loc='upper right')
    
    # Show plot
    plt.show()
    
    no_vote = [buff for buff in herd if buff.vote == 0]
    vote_a = [buff for buff in herd if buff.vote == 1]
    vote_b = [buff for buff in herd if buff.vote == 2]
    print(f'Tally:\nNo vote : {len(no_vote)}\nDirection A : {len(vote_a)}\nDirection B : {len(vote_b)}')
    
    # Extract positions into a numpy array
    # Build a k-d tree
    tree = cKDTree(herd_positions)
    
    # Find all pairs within range r
    neighbours = tree.query_pairs(r)
    
    # Create a networkx graph
    network_buffalos = nx.Graph()
    
    # Add all buffaloes as nodes (even if they have no neighbors)
    for i in range(len(herd)):
        network_buffalos.add_node(i)  # Add node for each buffalo
    
    # Add edges (neighbours) to the graph
    network_buffalos.add_edges_from(neighbours)
    
    # Draw the network
    node_colors = [color_map[herd[i].vote] for i in range(len(herd))]
    nx.draw(network_buffalos, arrows=True, with_labels=False, node_color=node_colors, 
            node_size=20, edge_color='black')
    plt.title('Buffalo Network')
    plt.show()
    
    # Get the largest connected component and its size
    largest_cc = max(nx.connected_components(network_buffalos), key=len)
    largest_cc_size = len(largest_cc)
    cc_ratio = largest_cc_size / herd_size * 100
    
    print(f'The largest connected subgroup is made of {largest_cc_size} buffaloes \nBased in this, {cc_ratio:.0f}% of the herd is connected')

simulate_herd(r=100, herd_size=350)