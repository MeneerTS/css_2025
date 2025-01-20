import numpy as np
class Herd:
    # A Herd has a list of bizons
    def __init__(self):
        self.bizons = []
    def add_bizon(self,bizon):
        #not necessary but convenient function
        self.bizons.append(bizon)
    def as_numpy(self):
        locations = []
        directions = []
        for bizon in self.bizons:
            locations.append(list(bizon.position))
            directions.append(list(bizon.direction))
        return np.array(locations), np.array(directions)

class Bizon:  
    #Bizons can have many properties!
    #Perhaps these properties can be used in voting
    #only position and direction are currently used in the simulation
    def __init__(self,position,direction):
        self.position = position
        self.direction = direction
        #self.hunger, self.size etc etc...