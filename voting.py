from herd import Herd, Bizon
import numpy as np

def random_herd(n,boundary_size):
    # Gives a Herd with n Bizons
    # with a random x and y pos < boundary_size
    # and a random unit vector direction
    this_herd = Herd()
    for i in range(n):
        location = np.random.rand(2)*boundary_size   #random pos between 0 and size
        direction = np.random.randn(2)      # unit vector
        b = Bizon(location,direction)
        this_herd.add_bizon(b)
    return this_herd
