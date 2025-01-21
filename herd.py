import numpy as np
class Herd:
    # A Herd has a list of bisons
    def __init__(self):
        self.bisons = []
    def add_bison(self, bison):
        #not necessary but convenient function
        self.bisons.append(bison)
    def as_numpy(self):
        locations = []
        directions = []
        for bison in self.bisons:
            locations.append(list(bison.position))
            directions.append(list(bison.direction))
        return np.array(locations), np.array(directions)

class Bison:  
    #bisons can have many properties!
    #Perhaps these properties can be used in voting
    #only position and direction are currently used in the simulation
    def __init__(self,position,direction):
        self.position = position
        self.direction = direction
        #self.hunger, self.size etc etc...