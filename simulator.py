import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from herd import Bison, Herd
from voting import random_herd


class Simulator:
    """
    We forgo defining birds as storing their information
    as numpy arrays allows us to work with them more efficiently

    We also change the way we think about location, velocity and accelerations
    to a direction approach. This almost completely is implemented the same way.
    Except for when the direction and the desired direction are combined. 
    We would like to combine them in such a way that the birds have a turning radius,
    since most birds can not fly backward. Modeling with pure acceleration would be more
    akin to drones.


    """
    def __init__(self, herd):
        
                                       
        self.boundary_size = 200 
        self.locations, self.directions = herd.as_numpy()
        self.n = len(self.locations)                            #number of birds
        
        self.desireddirection = np.zeros((self.n,2))     # unit vector
        self.eps = 2.                              # higher is more efficient neighbour search, but also extends the neighbour range randomly for some birds
        self.r = 20                                 # range that animals see eachother
        self.r_close = 7.5                           # range that animals try to avoid eachother
        self.r_close = self.r_close**2              # squared for convenient computation                            
        self.seperation = 8.
        self.alignment = 2.
        self.cohesion = 1.

    def find_pairs(self,r):
        #returns indexes, only one way around 
        #So [(animal1, animal2),(animal1,animal3),(animal2,animal4)...]
        tree = KDTree(self.locations,compact_nodes=True,balanced_tree=True)
        pairs = tree.query_pairs(r,2.0,self.eps,'ndarray')
        return pairs

    def get_neigbours(self,r):
        #Constructs a dictionary that to easily look up the neighbours of a certain animal
        # key animal_id, item a list of neighbours
        pairs = self.find_pairs(r)
        output = defaultdict(list)

        for item1, item2 in pairs:
            output[item1].append(item2)
            output[item2].append(item1)

        return output
    
    

    def apply_rules(self):
        neigbourdict = self.get_neigbours(self.r)
        accelerations = np.zeros((self.n,2))

        #Hmm, probably possible to optimize this
        #If you have a matrix of nxn with which bird is near which bird

        #Turns out! No! The fact that every animal has few neighbours (compared to the 1000 other animals) makes it not worth
        #vectorizing everything, dictionaries rule! 
        for boid in neigbourdict.keys():
            
            #Try to head the same way as the flock
            headings = self.directions[neigbourdict[boid]]
            accelerations[boid] += np.mean(headings,axis=0) * self.alignment 
            
            
            #Go towards the flock
            neigbour_locations = self.locations[neigbourdict[boid]]
            accelerations[boid] += np.mean(neigbour_locations - self.locations[boid],axis=0) * self.cohesion
            

            #at each neighbours index it says 1 or 0 if it is close or not
            close_neigbours = np.where(np.sum((neigbour_locations - self.locations[boid]) ** 2, axis=1) < self.r_close,True,False)            
            close_locations = neigbour_locations[close_neigbours]
            if len(close_locations) > 0:
                #Steer away from very close neighbours
                #Here we take the mean
                #This is equivalent to steering away from every neighbour seperatly (and scaling down)
                accelerations[boid] += np.mean(self.locations[boid] - close_locations, axis=0) * self.seperation
        #Normalise
        self.desireddirections = accelerations / (np.expand_dims(np.linalg.norm(accelerations,axis=1),1) + 0.000001)
    

    def update(self):
        self.apply_rules()        
        
        #Make it so the animals can only turn a certain amount
        #They can not just turn around suddenly
        self.directions = self.directions * 5 + self.desireddirections 
        
        #Make sure we always have unit speed and just change direction
        #Animals do not go backwards anyway
        self.directions = self.directions / (np.expand_dims(np.linalg.norm(self.directions,axis=1),1) + 0.000001)
        
        self.locations += self.directions   
        self.locations %= self.boundary_size
       


if __name__ == "__main__":

    this_herd = random_herd(50,100)
    this_sim = Simulator(this_herd)
    #print(flock.directions)
    #print(flock.desireddirection)
    fig, ax = plt.subplots()
    ax.set(xlim=[0, 200], ylim=[0, 200])
    scat = ax.scatter(this_sim.locations.T[0], this_sim.locations.T[1], c="b", s=5)

    def draw_boids(frame):
        this_sim.update()
        scat.set_offsets(this_sim.locations)


    #print(flock.locations)
    #print(flock.find_pairs(30))

    animation = FuncAnimation(fig, draw_boids, frames=50, interval=30)
    plt.show()