import numpy as np
class Herd:
    
    def __init__(self):
        self.bizons = []
    def add_bizon(self,bizon):
        self.bizons.append(bizon)
    def as_numpy(self):
        locations = []
        directions = []
        for bizon in self.bizons:
            locations.append(list(bizon.position))
            directions.append(list(bizon.direction))
        return np.array(locations), np.array(directions)

class Bizon:  
    def __init__(self,position,direction):
        self.position = position
        self.direction = direction
        #self.hunger etc etc...