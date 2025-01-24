import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from herd import Bison, Herd
from voting import random_vote

"""Some updates to the simulator.py code. The herd now has new behavior weights which incentivize the bisons 
to follow the group 85% of the time, and 15% of the time wander off. Now 45% of the bisons intially votes
and 55% of the bisons are supposed to make their direction decision based on their 10 nearest neighbours.
The rest of the dynamics should be the same (aka the incentives are the same)"""

class Simulator:
    def __init__(self, herd, boundary_size=1000):
        self.boundary_size = boundary_size
        self.locations, self.directions, self.votes = herd.as_numpy()
        self.n = len(self.locations)                            #number of birds
        
        self.desireddirection = np.zeros((self.n, 2))    # unit vector
        self.eps = 2.0                              # higher is more efficient neighbour search, but also extends the neighbour range randomly for some birds
        self.r = 20                                  # range that animals see eachother
        self.r_close = 7.5                           # range that animals try to avoid eachother
        self.r_close = self.r_close**2              # squared for convenient computation     

        self.seperation = 2.0
        self.alignment = 2.0
        self.cohesion = 1.0
        self.vote_bias = 1.0
        self.momentum_bias = 2.0

        #new added weights
        self.herd_follow_weight = 0.85  
        self.split_weight = 0.15  

        #voters -> 45% vote, 55% don't
        self.voters = np.random.choice([True, False], size=self.n, p=[0.45, 0.55])
        self.votes = np.zeros((self.n, 2))  
        
        self.votes[self.voters] = np.random.randn(np.sum(self.voters), 2)
        self.votes[self.voters] /= np.linalg.norm(self.votes[self.voters], axis=1)[:, None]  # Normalize vote vectors

    def find_pairs(self, r):
        #returns indexes, only one way around 
        #so [(animal1, animal2),(animal1,animal3),(animal2,animal4)...]
        tree = KDTree(self.locations, compact_nodes=True, balanced_tree=True)
        pairs = tree.query_pairs(r, 2.0, self.eps, "ndarray")
        return pairs

    def get_neighbours(self, r):
        #constructs a dictionary that to easily look up the neighbours of a certain animal
        #key animal_id, item a list of neighbours
        pairs = self.find_pairs(r)
        output = defaultdict(list)
        for item1, item2 in pairs:
            output[item1].append(item2)
            output[item2].append(item1)
        return output

    def apply_rules(self):
        neighbour_dict = self.get_neighbours(self.r)
        accelerations = np.zeros((self.n, 2))

        for bison in neighbour_dict.keys():
            
            headings = self.directions[neighbour_dict[bison]]
            #allignment
            mean_heading = np.mean(headings, axis=0)
            accelerations[bison] += mean_heading / (np.linalg.norm(mean_heading) + 0.000001) * self.alignment

            #cohesion -> move towards the average position of neighbors
            neighbour_locations = self.locations[neighbour_dict[bison]]
            accelerations[bison] += np.mean(neighbour_locations - self.locations[bison], axis=0) * self.cohesion

            #separation -> move away from nearby neighbors to avoid crowding
            close_neigbours = np.where(np.sum((neighbour_locations - self.locations[bison]) ** 2, axis=1) < self.r_close, True, False)
            close_locations = neighbour_locations[close_neigbours]
            if len(close_locations) > 0:
                
                accelerations[bison] += np.average(
                    self.locations[bison] - close_locations,
                    axis=0,
                    weights=(1 / np.linalg.norm(self.locations[bison] - close_locations, axis=1))
                ) * self.seperation

            #voting
            if self.voters[bison]:
                #so, bisons that voted move based on their vote direction
                accelerations[bison] += self.votes[bison] * self.vote_bias
            else:
                #bisons that didn´t vote move based on 10 of their closest voters' directions
                close_voters = [n for n in neighbour_dict[bison] if self.voters[n]]
                if len(close_voters) > 0:
                    closest_voters = sorted(close_voters, key=lambda n: np.linalg.norm(self.locations[bison] - self.locations[n]))[:10]
                    voter_directions = self.votes[closest_voters]
                    mean_voter_direction = np.mean(voter_directions, axis=0)
                    accelerations[bison] += mean_voter_direction / (np.linalg.norm(mean_voter_direction) + 0.000001) * self.vote_bias

            #implementing wandering behavior (15% chance)
            if np.random.rand() < self.split_weight:  
                #random direction to the wandering direction
                random_direction = np.random.randn(2)
                random_direction /= np.linalg.norm(random_direction)  
                accelerations[bison] += random_direction * self.seperation  

        
        self.desireddirections = accelerations / (np.expand_dims(np.linalg.norm(accelerations, axis=1), 1) + 0.000001)

    def update(self):
        #update the bisons' positions and directions
        self.apply_rules()

        self.directions = self.directions * self.momentum_bias + self.desireddirections
        self.directions = self.directions / (np.expand_dims(np.linalg.norm(self.directions, axis=1), 1) + 0.000001)
        self.locations += self.directions
        self.locations %= self.boundary_size  

if __name__ == "__main__":
    grid_size = 200
    this_herd = Herd(90, [grid_size // 2, grid_size // 2], 12)
    random_vote(this_herd) 

    this_sim = Simulator(this_herd, grid_size)

    fig, ax = plt.subplots()
    ax.set(xlim=[0, grid_size], ylim=[0, grid_size])

    herd_center_x = []
    herd_center_y = []
    trace = ax.plot([], [], c="blue")

    scat = ax.scatter(this_sim.locations.T[0], this_sim.locations.T[1], c="brown", s=5)

    def draw_bisons(frame):
        this_sim.update()
        scat.set_offsets(this_sim.locations)
        herd_center = np.mean(this_sim.locations, axis=0)
        herd_center_x.append(herd_center[0])
        herd_center_y.append(herd_center[1])
        trace[0].set_data(herd_center_x, herd_center_y)

    avg_vote = np.mean(this_sim.votes, axis=0)
    avg_vote /= np.linalg.norm(avg_vote)
    avg_vote *= grid_size // 3

    plt.quiver(grid_size // 2, grid_size // 2, avg_vote[0], avg_vote[1], angles="xy", scale_units="xy", scale=1, color="red", alpha=0.6)

    animation = FuncAnimation(fig, draw_bisons, frames=50, interval=200)
    plt.show()
