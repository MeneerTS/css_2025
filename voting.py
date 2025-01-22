import numpy as np

def random_vote(herd):
    for bison in herd.bisons:
        direction = np.random.rand(2)
        direction /= np.linalg.norm(direction)
        bison.direction = direction
