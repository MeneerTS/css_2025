import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sim import Sim
import voting
import numpy as np

def plot_voting(sim: Sim, show = True) -> tuple:
    """
    Plots colored points based on the votes of the bisons and their locations

    Parameters:
    sim: Sim - the simulation object
    show: bool - whether to show the plot

    Returns:
    fig, ax, scat: tuple - the figure, axis and scatter objects
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set(xlim=[0, 200], ylim=[0, 200])
    colors = ["blue" if vote[0] == 1 else "red" if vote[1] == 1 else "grey" for vote in sim.votes]
    scat = ax.scatter(sim.locations[:, 0], sim.locations[:, 1], c=colors, s=5, zorder=1)
    if show:
        plt.show()
    return fig, ax, scat


def plot_spheres_of_influence(sim: Sim, r: float, show = True) -> tuple:
    """
    Plots colored points based on the votes of the bisons and their locations, and plot circles
    around the voting bisons even if the unvoting bisons have voted through some other means (i.e.
    influence from neighbours).

    Parameters:
    sim: Sim - the simulation object
    r: float - the radius of the circles
    show: bool - whether to show the plot

    Returns:
    fig, ax: tuple - the figure and axis objects
    """
    fig, ax, _ = plot_voting(sim, False)
    for locationid in range(sim.num_voters):
        circle = plt.Circle(sim.locations[locationid], r, fill=False)
        ax.add_patch(circle)
    if show:
        plt.show()
    return fig, ax


def r_analysis(num_r_values: int, num_voters: int, num_non_voters: int, num_iterations_per_r: int) -> None:
    """
    Analyze the effect of the radius r on the voting outcome.

    Parameters:
    num_r_values: int - the number of r values to analyze
    num_voters: int - the number of voters
    num_non_voters: int - the number of non-voters
    num_iterations_per_r: int - the number of iterations per r

    """
    total_results = []
    all_data = []

    total_cows = num_voters + num_non_voters
    sim = Sim(total_cows, num_voters)

    field_radius = 50*np.sqrt((total_cows)/100)
    

    for r_value in range(0, 100, 100//num_r_values):
        print(f"---\n{r_value}---\n")
        results = []
        sim.reset(total_cows, num_voters, radius=field_radius)
        for _ in range(num_iterations_per_r):
            r = r_value
            sim.reset(total_cows, num_voters, radius=field_radius)
            points_within_range = voting.simulate_voting(sim, r, 0.5)
            # print(voting.get_majority(sim))
            voting.unvoted_vote(sim,points_within_range)
            mean_votes = np.mean(sim.votes,axis=0)
            results.append(np.abs(mean_votes[0] - mean_votes[1]))
        total_results.append(np.mean(results))
        all_data.append(results)

    # Check that the number of results is correct
    assert len(total_results) == num_r_values

    # Save data to numpy files
    np.save(f"results/voters_{num_voters}_non_voters_{num_non_voters}_number_of_r_\
            {num_r_values}_iterations_per_r_{num_iterations_per_r}_r_analysis_plot",total_results)
    np.save(f"results/voters_{num_voters}_non_voters_{num_non_voters}_number_of_r_\
            {num_r_values}_iterations_per_r_{num_iterations_per_r}_r_analysis_all_data",all_data)

    _, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ax.plot(np.arange(0, 100, 100//num_r_values), total_results)

    plt.show()


def histogram_analysis(r: float) -> None:
    """
    Show a histogram of the voting results after 1000 iterations for a given radius r.

    Parameters:
    r: float - the radius within which to look for neighbours
    """
    results = []
    for _ in range(1000):
        sim = Sim()
        points_within_range = voting.simulate_voting(sim, r)
        voting.unvoted_vote(sim, points_within_range)
        mean_votes = np.mean(sim.votes, axis=0)
        results.append(mean_votes[0])

    _, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(results, bins=25)

    plt.show()


def influence_analysis(steps: int) -> None:

    total_results_r = []
    sim = Sim(100, 100)
    for r in [0, 25, 50, 75, 100]:
        total_results = []
        all_data = []
        sim.reset(100, 100)
        for pb_value in np.linspace(0.4, 0.6, steps):
            results = []
            sim.reset(100, 100)
            print(f"r: {r}, pb: {pb_value}")
            for _ in range(1000):
                sim.reset(100, 100)
                points_within_range = voting.simulate_voting(sim, r, pb_value)
                voting.unvoted_vote(sim, points_within_range)
                results.append(1 if (np.array_equal(voting.get_majority(sim), np.array([1, 0]))) else 0)
            total_results.append(np.mean(results))
            all_data.append(results)

        # Check that the number of results is correct
        assert len(total_results) == steps

        # Save data to numpy files
        np.save(f"results/r_{r}_steps_{steps}_influence_analysis_plot", total_results)
        np.save(f"results/r_{r}_steps_{steps}_influence_analysis_all_data", all_data)
        total_results_r.append(total_results)

    _, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    for total_results in total_results_r:
        ax.plot(np.linspace(0.4, 0.6, steps), total_results)
    ax.legend(["r = 0", "r = 25", "r = 50", "r = 75", "r = 100"])
    ax.set_xlabel("Bias of voting blue")
    ax.set_ylabel("Percentage of times majority is blue")

    plt.show()


def uninformed_analysis() -> None:
    total_results_r = []
    sim = Sim(100, 100)
    for r in [10, 12, 14, 16, 18, 20]:
        total_results = []
        all_data = []
        sim.reset(100, 100)
        for num_nonvoters in np.linspace(0, 100, 21):
            results = []
            # sim.reset(100, 100)
            print(f"r: {r}, num_nonvoters: {num_nonvoters}")
            for _ in range(1000):
                sim.reset(100, num_nonvoters)
                points_within_range = voting.simulate_voting(sim, r, 0.5)
                first_majority = voting.get_majority(sim)
                if first_majority[0] == 0 and first_majority[1] == 0:
                    results.append(0.5)
                    continue
                voting.unvoted_vote(sim, points_within_range, num_nonvoters)
                majority = voting.get_majority(sim)
                if np.array_equal(first_majority, majority):
                    results.append(1)
                else:
                    results.append(0)
            total_results.append(np.mean(results))
            all_data.append(results)
        
        # Check that the number of results is correct
        assert len(total_results) == 21
    
        # Save data to numpy files
        np.save(f"results/r_{r}_uninformed_analysis_plot", total_results)
        np.save(f"results/r_{r}_uninformed_analysis_all_data", all_data)
        total_results_r.append(total_results)

    _, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    for total_results in total_results_r:
        ax.plot(np.linspace(0, 100, 21), total_results)
    ax.legend(["r = 10", "r = 12", "r = 14", "r = 16", "r = 18", "r = 20"])
    ax.set_xlabel("Number of non-voters")
    ax.set_ylabel("Percentage of times majority is blue")

    plt.show()



def run_animation(r: float) -> None:
    """
    Run an animation of the voting process for a given radius r.

    Parameters:
    r: float - the radius within which to look for neighbours
    """

    sim = Sim()

    fig, ax, scat = plot_voting(sim, False)
    circle = plt.Circle(sim.locations[0], r, fill=False)
    ax.add_patch(circle)


    points_within_range = sim.compute_neighbours(r)
    print(points_within_range)
    # Print avg number of neighbours
    print(np.mean([len(neighbors) for neighbors in points_within_range]))

    # Create lines between points within range
    for i, neighbors in enumerate(points_within_range):
        for j in neighbors:
            ax.plot([sim.locations[i][0], sim.locations[j][0]],
                    [sim.locations[i][1], sim.locations[j][1]],
                    c="gray", alpha=1, zorder=0)

    def update(_):
        if sim.num_voted == sim.num_voters:
            print("All inital votes cast")

            # Place the circle around all voters
            circle.center = [100, 100]
            circle.radius = 50
            sim.num_voted += 1

            # Check if all bisons still exist
            assert sim.herd_size == len(sim.herd.bisons)

        if sim.num_voted < sim.num_voters:
            # Display the radius of the current voter
            circle.center = sim.locations[sim.num_voted]

            # Tallies all local votes
            votes_neighbours = [vote for vote in sim.votes[points_within_range[sim.num_voted]]
                                if vote[0] == 1 or vote[1] == 1]

            # Compute the influence of the neighbours
            influence = np.mean(votes_neighbours, axis=0) if len(votes_neighbours) > 0 else np.array([0, 0])
            influence = (influence[0] - influence[1]) / 2

            # Influence should be between -0.5 and 0.5
            assert -0.5 <= influence <= 0.5

            voting.random_vote_indexed(sim.herd, sim.num_voted, 0.5 + influence)
            sim.locations, sim.votes = sim.herd.as_numpy()
            sim.num_voted += 1
            colors = ["blue" if vote[0] == 1 else "red" if vote[1] == 1 else "grey" for vote in sim.votes]
            scat.set_facecolor(colors)

    _ = FuncAnimation(fig, update, frames=50, interval=200)

    plt.show()



if __name__ == "__main__":
    
    #print(np.mean(results))
    # run_animation(10)
    r_analysis(10, 400, 0, 100)
    # histogram_analysis(15)
    # influence_analysis(20)
    #uninformed_analysis()

