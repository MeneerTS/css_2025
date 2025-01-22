import numpy as np
class Herd:
    # A Herd has a list of bisons
    def __init__(self, count = 0, center = [100, 100], radius = 50):
        self.bisons = []
        for _ in range(count):
            r = radius * np.sqrt(np.random.rand(count))
            theta = 2 * np.pi * np.random.rand(count)
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            positions = np.stack((x, y), axis=-1)
            self.bisons = [Bison(pos, np.zeros(2)) for pos in positions]


    def add_bison(self, bison):
        #not necessary but convenient function
        self.bisons.append(bison)
    def as_numpy(self):
        locations = []
        directions = []
        votes = []
        for bison in self.bisons:
            locations.append(list(bison.position))
            directions.append(list(bison.direction))
            votes.append(list(bison.vote))
        return np.array(locations), np.array(directions), np.array(votes)

class Bison:  
    #bisons can have many properties!
    #Perhaps these properties can be used in voting
    #only position and direction are currently used in the simulation
    def __init__(self,position, direction):
        self.position = position
        self.direction = direction
        self.vote = direction
        #self.hunger, self.size etc etc...