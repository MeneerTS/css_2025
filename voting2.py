import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

num_bisons = 100  
initial_voter_fraction = 0.35    
initial_radius = 0.1    # initial influence radius
radius_step = 0.05      #step to increase radius for isolated bisons 
iterations = 10

positions = np.random.rand(num_bisons, 2)

#initializes votes (-1 for undecided, 0-3 for directions)
votes = -1 * np.ones(num_bisons, dtype=int)
influence_weights = np.ones(num_bisons, dtype=float)  

#35% of the bisons vote
num_initial_voters = int(initial_voter_fraction * num_bisons)
initial_voters = np.random.choice(num_bisons, num_initial_voters, replace=False)

for voter in initial_voters:
    votes[voter] = np.random.choice([0, 1, 2, 3])  #random direction
    
#finds neighbors within a given radius
def find_neighbors(positions, index, radius):
    distances = np.linalg.norm(positions - positions[index], axis=1)
    neighbors = np.where(distances < radius)[0]
    neighbors = neighbors[neighbors != index] 
    return neighbors

#updates votes based on neighbors
def update_votes(positions, votes, influence_weights, radius):
    new_votes = votes.copy()
    undecided = np.where(votes == -1)[0]
    for i in undecided:
        neighbors = find_neighbors(positions, i, radius)
        neighbor_votes = votes[neighbors]
        neighbor_weights = influence_weights[neighbors]
        voted_neighbors = neighbor_votes[neighbor_votes != -1]  #excludes undecided neighbors
        if len(voted_neighbors) > 0:
            #checks if there's an influential neighbor
            if np.any(neighbor_weights[neighbor_votes != -1] > 1):
                influential_vote = neighbor_votes[np.argmax(neighbor_weights)]
                new_votes[i] = influential_vote
            else:
                #computes the weighted majority vote
                unique, counts = np.unique(voted_neighbors, return_counts=True)
                weighted_counts = [np.sum(neighbor_weights[neighbor_votes == u]) for u in unique]
                new_votes[i] = unique[np.argmax(weighted_counts)]
    return new_votes

def visualize(positions, votes, title, connections):
    plt.figure(figsize=(8, 8))
    colors = ["blue", "red", "green", "orange", "gray"]  

    #draws connections between undecided bisons and their voting neighbors
    for conn in connections:
        bison, neighbor = conn
        plt.plot([positions[bison][0], positions[neighbor][0]],
                 [positions[bison][1], positions[neighbor][1]], color="black", alpha=0.5, linestyle="--")

    for i, (x, y) in enumerate(positions):
        color = "gray" if votes[i] == -1 else colors[votes[i]]
        plt.scatter(x, y, color=color, s=50, label=f"Direction {votes[i]}" if votes[i] != -1 else "Undecided")

    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()
    plt.show()

#loop
radius = initial_radius
connections = []
visualize(positions, votes, "Initial Votes", connections)
for iteration in range(iterations):
    connections = []
    undecided = np.where(votes == -1)[0]
    for i in undecided:
        neighbors = find_neighbors(positions, i, radius)
        for neighbor in neighbors:
            if votes[neighbor] != -1:
                connections.append((i, neighbor))

    votes = update_votes(positions, votes, influence_weights, radius)
    visualize(positions, votes, f"Iteration {iteration + 1}", connections)

    if np.all(votes != -1):
        print(f"All bisons voted by iteration {iteration + 1}")
        break

    #increases radius for isolated bisons if there are any
    if np.any(votes == -1):
        radius += radius_step

#makes all bisons converge on the unanimous decision
unique, counts = np.unique(votes, return_counts=True)
weighted_counts = [np.sum(influence_weights[votes == u]) for u in unique]
final_vote = unique[np.argmax(weighted_counts)]
votes[:] = final_vote  

connections = [(i, j) for i in range(num_bisons) for j in find_neighbors(positions, i, radius)]
visualize(positions, votes, "Final Convergence: Unanimous Decision", connections)
