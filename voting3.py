import numpy as np
import matplotlib.pyplot as plt

num_bisons = 100 
initial_voter_fraction = 0.35 
initial_radius = 0.1  #initial influence radius
radius_step = 0.05  #step to increase radius for isolated bisons
iterations = 10  #number of iterations

#initializes bison positions randomly
positions = np.random.rand(num_bisons, 2)

votes = -1 * np.ones(num_bisons, dtype=int)

#randomly assign initial voters
num_initial_voters = int(initial_voter_fraction * num_bisons)
initial_voters = np.random.choice(num_bisons, num_initial_voters, replace=False)
for voter in initial_voters:
    votes[voter] = np.random.choice([0, 1, 2, 3])

#finds neighbors within a given radius
def find_neighbors(positions, index, radius):
    distances = np.linalg.norm(positions - positions[index], axis=1)
    neighbors = np.where(distances < radius)[0]
    neighbors = neighbors[neighbors != index]  
    return neighbors

#updates votes based on majority rule
def update_votes(positions, votes, radius):
    new_votes = votes.copy()
    undecided = np.where(votes == -1)[0]  
    for i in undecided:
        neighbors = find_neighbors(positions, i, radius)
        neighbor_votes = votes[neighbors]
        voted_neighbors = neighbor_votes[neighbor_votes != -1]  
        if len(voted_neighbors) > 0:
            #computes majority votes
            unique, counts = np.unique(voted_neighbors, return_counts=True)
            new_votes[i] = unique[np.argmax(counts)]
    return new_votes

def visualize(positions, votes, title, connections):
    plt.figure(figsize=(8, 8))
    colors = ["blue", "red", "green", "orange", "gray"]  

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

    votes = update_votes(positions, votes, radius)
    visualize(positions, votes, f"Iteration {iteration + 1}", connections)

    if np.all(votes != -1):
        print(f"All bisons voted by iteration {iteration + 1}")
        break

    #increase radius for isolated bisons if any remain undecided
    if np.any(votes == -1):
        radius += radius_step

#unanimous decision (majority rule across the herd)
unique, counts = np.unique(votes, return_counts=True)
final_vote = unique[np.argmax(counts)]
votes[:] = final_vote  #all bisons adopt the global vote

connections = [(i, j) for i in range(num_bisons) for j in find_neighbors(positions, i, radius)]
visualize(positions, votes, "Final Convergence: Unanimous Decision", connections)
