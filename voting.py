import numpy as np

def random_vote(herd):
    for bison in herd.bisons:
        vote = np.random.rand(2) * 2 - 1
        vote /= np.linalg.norm(vote)
        bison.vote = vote
