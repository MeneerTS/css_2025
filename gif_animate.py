import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import argparse
from sim import Sim
from loading_and_plotting import plot_voting
import voting

def run_animation(r: float, save_path="voting_animation.gif") -> None:
    """
    Runs the animation of the voting process for a given radius r and saves it as a GIF.
    
    Parameters:
    r: float
        The radius within which to look for neighbours.
    save_path: str
        The file path where the GIF will be saved.
    """
    sim = Sim(50, 40)

    fig, ax, scat = plot_voting(sim, False)
    fig.patch.set_facecolor('#D2B48C')
    
    circle = plt.Circle(sim.locations[0], r, fill=False)
    ax.add_patch(circle)

    #Computes neighbours and prints the average number of neighbours
    points_within_range = sim.compute_neighbours(r)
    print("Average number of neighbours:", np.mean([len(neighbors) for neighbors in points_within_range]))

    #Draws the lines between points within range 
    lines = []
    for i, neighbors in enumerate(points_within_range):
        for j in neighbors:
            line, = ax.plot([sim.locations[i][0], sim.locations[j][0]],
                            [sim.locations[i][1], sim.locations[j][1]],
                            c="gray", alpha=0.2, zorder=0)
            lines.append((i, j, line))

    def update(frame):
        if sim.num_voted == sim.num_voters:
            print("All inital votes cast")

            #place the circle around all voters
            circle.center = [100, 100]
            circle.radius = 50
            sim.num_voted += 1

            #Check that all bisons still exist
            assert sim.herd_size == len(sim.herd.bisons)

        if sim.num_voted < sim.num_voters:
            #Updates circle position to the current voter
            circle.center = sim.locations[sim.num_voted]

            #Tallies all local votes
            votes_neighbours = [vote for vote in sim.votes[points_within_range[sim.num_voted]]
                                if vote[0] == 1 or vote[1] == 1]

            # Compute influence from neighbours
            influence = np.mean(votes_neighbours, axis=0) if len(votes_neighbours) > 0 else np.array([0, 0])
            influence = (influence[0] - influence[1]) / 2

            # Influence should be within -0.5 to 0.5
            assert -0.5 <= influence <= 0.5

            voting.random_vote_indexed(sim.herd, sim.num_voted, 0.5 + influence)
            sim.locations, sim.votes = sim.herd.as_numpy()
            sim.num_voted += 1
            colors = ["blue" if vote[0] == 1 else "red" if vote[1] == 1 else "grey" for vote in sim.votes]
            scat.set_facecolor(colors)

        #Dynamically updates the drawn lines in case positions have changed
        for i, j, line in lines:
            line.set_data(
                [sim.locations[i][0], sim.locations[j][0]],
                [sim.locations[i][1], sim.locations[j][1]]
            )

    #Creates the animation
    anim = FuncAnimation(fig, update, frames=50, interval=200)

    #Saves the animation as a GIF using PillowWriter (the fps can be changes if it´s too slow or fast)
    anim.save(save_path, writer=PillowWriter(fps=5))
    print(f"Animation saved as {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the voting animation and save it as a GIF.")
    parser.add_argument("r", type=float, help="The radius within which to look for neighbours.")
    parser.add_argument("--save_path", type=str, default="voting_animation.gif",
                        help="The file path where the GIF will be saved.")
    args = parser.parse_args()

    run_animation(args.r, args.save_path)
