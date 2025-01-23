import random
import numpy as np
import herd

def random_vote(herd, count, p = 0.5):
    for i, bison in enumerate(herd.bisons):
        if i < count:
            if random.random() < p:
                bison.vote = np.array([1, 0])
            else:
                bison.vote = np.array([0, 1])
        else:
            bison.vote = np.array([0, 0])

def random_vote_indexed(herd, i, p = 0.5):
    if random.random() < p:
        herd.bisons[i].vote = np.array([1, 0])
    else:
        herd.bisons[i].vote = np.array([0, 1])
