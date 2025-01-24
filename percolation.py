import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from herd import Herd
import voting
import numpy as np
import copy

class Sim:
    def __init__(self):
        self.herd_size = 100
        self.herd = Herd(self.herd_size, [100, 100], 50)
        # This should create a herd of herd_size bisons
        assert len(self.herd.bisons) == self.herd_size

        self.locations, self.votes = self.herd.as_numpy()
        # Every bison should have a location and a vote
        assert len(self.locations) == self.herd_size and len(self.votes) == self.herd_size

        self.num_voted = 0
        # The number of voters should start at 0
        assert self.num_voted == 0

        self.num_voters = 50
        # Only a total of herd_size bisons can vote
        assert self.num_voters <= self.herd_size



    def compute_neighbours(self, r):
        # Compute pairwise distances
        pairwise_distances = np.linalg.norm(self.locations[:, np.newaxis] - self.locations, axis=2)
        print(pairwise_distances)

        # Find points within the range for each point
        points_within_range = [np.where((pairwise_distances[i] <= r) & (pairwise_distances[i] != 0))[0] for i in range(len(self.locations))]
        print(points_within_range)
        # Display results
        for i, neighbors in enumerate(points_within_range):
            print(f"Points within range of {i}: {neighbors}")

        return points_within_range

def simulate_voting(sim,r):
    #simulates a voting cycle without making any pictures
    points_within_range = sim.compute_neighbours(r)
    while True:
        if sim.num_voted >= sim.num_voters:
            return
        if sim.num_voted < sim.num_voters:
            votes_neighbours = [vote for vote in sim.votes[points_within_range[sim.num_voted]] if vote[0] == 1 or vote[1] == 1]
            influence = np.mean(votes_neighbours, axis=0) if len(votes_neighbours) > 0 else np.array([0, 0])
            #print(influence)
            influence = (influence[0] - influence[1]) / 2
            # influence = 0
            assert -0.5 <= influence <= 0.5
                
            voting.random_vote_indexed(sim.herd, sim.num_voted, 0.5 + influence)
            sim.locations, sim.votes = sim.herd.as_numpy()
            sim.num_voted += 1
def unvoted_vote(sim,r): 
    points_within_range = sim.compute_neighbours(r)
    #The unvoted bizons go with the majority in their radius
    

    for unvotedindex in range(sim.num_voted,sim.herd_size):
        votes_neighbours = [vote for vote in sim.votes[points_within_range[unvotedindex]] if vote[0] == 1 or vote[1] == 1]
        influence = np.mean(votes_neighbours, axis=0) if len(votes_neighbours) > 0 else np.array([0, 0])
        #print(influence)
        #influence = (influence[0] - influence[1]) / 2
        #assert -0.5 <= influence <= 0.5
        if len(votes_neighbours) > 0:
            #If there is a neighbour we look
            #At which has the most influence
            #and do that vote
            influence = influence[0] > influence[1]
            voting.random_vote_indexed(sim.herd, unvotedindex, influence)
        else:
            #Otherwise do random
            voting.random_vote_indexed(sim.herd, unvotedindex, 0.5)
    #update only at the end
    sim.locations, sim.votes = sim.herd.as_numpy()  
        
def plot_voting(sim,r):
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    ax.set_aspect('equal')
    ax.set(xlim=[0, 200], ylim=[0, 200])

    colors = ["blue" if vote[0] == 1 else "red" if vote[1] == 1 else "grey" for vote in sim.votes]
    #circle = plt.Circle(sim.locations[0], r, fill=False)
    #ax.add_patch(circle) 


    points_within_range = sim.compute_neighbours(r)

    # Create lines between points within range
    #for i, neighbors in enumerate(points_within_range):
    #    for j in neighbors:
    #        ax.plot([sim.locations[i][0], sim.locations[j][0]], [sim.locations[i][1], sim.locations[j][1]], c="gray", alpha=1, zorder=0)

    scat = ax.scatter(sim.locations[:, 0], sim.locations[:, 1], c=colors, s=5, zorder=1)
    plt.show()

def analyze_voting_thijs():
    
    r = 1
    sim = Sim()
    simulate_voting(sim,r)
    print(np.mean(sim.votes,axis=0))
    #pretty_pic(sim,r)
    unvoted_vote(sim,r)
    #pretty_pic(sim,r)
    print(np.mean(sim.votes,axis=0))
    

if __name__ == "__main__":
    
    # analyze_voting_thijs()
    # exit()
    r = 10
    sim = Sim()

    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    ax.set_aspect('equal')
    ax.set(xlim=[0, 200], ylim=[0, 200])

    colors = ["blue" if vote[0] == 1 else "red" if vote[1] == 1 else "grey" for vote in sim.votes]
    circle = plt.Circle(sim.locations[0], r, fill=False)
    ax.add_patch(circle) 


    points_within_range = sim.compute_neighbours(r)

    # Create lines between points within range
    for i, neighbors in enumerate(points_within_range):
        for j in neighbors:
            ax.plot([sim.locations[i][0], sim.locations[j][0]], [sim.locations[i][1], sim.locations[j][1]], c="gray", alpha=1, zorder=0)

    scat = ax.scatter(sim.locations[:, 0], sim.locations[:, 1], c=colors, s=5, zorder=1)

    def update(frame):
        if sim.num_voted == sim.num_voters:
            print("All inital votes cast")
            circle.center = [100, 100]
            circle.radius = 50
            sim.num_voted += 1
        if sim.num_voted < sim.num_voters:
            circle.center = sim.locations[sim.num_voted]
            votes_neighbours = [vote for vote in sim.votes[points_within_range[sim.num_voted]] if vote[0] == 1 or vote[1] == 1]
            influence = np.mean(votes_neighbours, axis=0) if len(votes_neighbours) > 0 else np.array([0, 0])
            print(influence)
            influence = (influence[0] - influence[1]) / 2
            # influence = 0
            assert -0.5 <= influence <= 0.5
                
            voting.random_vote_indexed(sim.herd, sim.num_voted, 0.5 + influence)
            sim.locations, sim.votes = sim.herd.as_numpy()
            sim.num_voted += 1
            colors = ["blue" if vote[0] == 1 else "red" if vote[1] == 1 else "grey" for vote in sim.votes]
            scat.set_facecolor(colors)
            print(sim.num_voted)

    animation = FuncAnimation(fig, update, frames=50, interval=200)

    plt.show()

