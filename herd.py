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
            directions = np.random.rand(count, 2)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            self.bisons = [Bison(pos, dir) for pos, dir in zip(positions, directions)]


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