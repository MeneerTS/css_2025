import numpy as np

class Buffalo:
    def __init__(self, x_position, y_position):
        self.x = x_position
        self.y = y_position
        self.position = [x_position, y_position]
        self.decision = None  # Initialize decision to None
        
        a = np.random.uniform(0, 1)
        if a > .45:
            self.vote = 0  # not an active voter
        else:
            b = np.random.choice([0, 1])
            if b == 1:
                self.vote = 1  # votes for A
            else:
                self.vote = 2  # votes for B