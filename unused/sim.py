import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from herd import Herd
from herd import Bison as Bison

class Sim:
    def __init__(self):
        self.herd = Herd( 50, [100, 100], 50)

    
    
if __name__ == "__main__":
    sim = Sim()
    fig, ax = plt.subplots()
    ax.set(xlim=[0, 200], ylim=[0, 200])
    locations, directions = sim.herd.as_numpy()
    print(locations)
    scat = ax.scatter(locations[:, 0], locations[:, 1], c="brown", s=5)

    def draw_bison(frame):
        pass
        # this_sim.update()
        # scat.set_offsets(this_sim.locations)


    #print(flock.locations)
    #print(flock.find_pairs(30))

    animation = FuncAnimation(fig, draw_bison, frames=50, interval=30)
    plt.show()

