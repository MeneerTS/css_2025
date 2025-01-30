import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.animation import FuncAnimation
from sim import Sim
import voting
import numpy as np
import argparse

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
    ax.set(xlim=[25, 175], ylim=[25, 175])
    ax.set_xticks([])
    ax.set_yticks([])
    colors = ["blue" if vote[0] == 1 else "red" if vote[1] == 1 else "grey" for vote in sim.votes]
    scat = ax.scatter(sim.locations[:, 0], sim.locations[:, 1], c=colors, s=30, zorder=1)
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
    _, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    for iterations in [0, 1, 2, 4]:
        total_results = []
        all_data = []

        total_cows = num_voters + num_non_voters
        sim = Sim(total_cows, num_voters)

        field_radius = 50*np.sqrt(total_cows/100)

        for r_value in range(0, 100, 100//num_r_values):
            print(f"---\n{r_value}---\n")
            results = []
            sim.reset(total_cows, num_voters, field_radius)
            for _ in range(num_iterations_per_r):
                r = r_value
                sim.reset(total_cows, num_voters, field_radius)
                points_within_range = voting.simulate_voting(sim, r, 0.5)
                # print(voting.get_majority(sim))
                voting.unvoted_vote(sim,points_within_range)
                for i in range(iterations):
                    voting.choose_local_majority(sim, points_within_range)
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

        x = np.arange(0, 100, 100//num_r_values)
        ax.plot(x, total_results)
        ax.set_xlabel("Vision range")
        ax.set_ylabel("Voter agreement")
        ax.grid(True)

    # , ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    for iterations in [1, 2, 4]:
        total_results = []
        all_data = []

        total_cows = num_voters + num_non_voters
        sim = Sim(total_cows, num_voters)

        field_radius = 50*np.sqrt(total_cows/100)

        for r_value in range(0, 100, 100//num_r_values):
            print(f"---\n{r_value}---\n")
            results = []
            sim.reset(total_cows, num_voters, field_radius)
            for _ in range(num_iterations_per_r):
                r = r_value
                sim.reset(total_cows, num_voters, field_radius)
                points_within_range = voting.simulate_voting(sim, r, 0.5)
                # print(voting.get_majority(sim))
                voting.unvoted_vote(sim,points_within_range)
                for i in range(iterations):
                    #randomize
                    sim.randomize_positions()
                    points_within_range = sim.compute_neighbours(r)
                    voting.choose_local_majority(sim, points_within_range)
                mean_votes = np.mean(sim.votes,axis=0)
                results.append(np.abs(mean_votes[0] - mean_votes[1]))
            total_results.append(np.mean(results))
            all_data.append(results)

        # Check that the number of results is correct
        assert len(total_results) == num_r_values

        # Save data to numpy files
        # np.save(f"results/voters_{num_voters}_non_voters_{num_non_voters}_number_of_r_\
                # {num_r_values}_iterations_per_r_{num_iterations_per_r}_r_analysis_plot",total_results)
        # np.save(f"results/voters_{num_voters}_non_voters_{num_non_voters}_number_of_r_\
                # {num_r_values}_iterations_per_r_{num_iterations_per_r}_r_analysis_all_data",all_data)

        if iterations == 1:
            ax.plot(x, total_results, linestyle="--", c="C1")
        # ax.plot(np.arange(0, 100, 100//num_r_values), total_results, linestyle="--")
        if iterations == 2:
            ax.plot(x, total_results, linestyle="--", c="C2")
        if iterations == 4:
            ax.plot(x, total_results, linestyle="--", c="C3")


    ax.legend(["0 iterations", "1 iteration", "2 iterations", "4 iterations", "1 iteration random", "2 iterations random", "4 iterations random"])
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

def iteration_analysis(num_voters:int, num_non_voters:int) -> None:

    total_cows = num_voters + num_non_voters
    sim = Sim(total_cows, num_voters)
    r_data = []

    _, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    for r_value in [10, 20]:
        print(f"---\n{r_value}---\n")
        
        sim.reset(total_cows, num_voters)
        run_values = []
        for _ in range(100):

            r = r_value
            sim.reset(total_cows, num_voters)
            results = []


            points_within_range = voting.simulate_voting(sim, r, 0.5)
            voting.unvoted_vote(sim,points_within_range)

            mean_votes = np.mean(sim.votes,axis=0)
            print(mean_votes)
            results.append(np.abs(mean_votes[0] - mean_votes[1]))

            for i in range(10):
                voting.choose_local_majority(sim, points_within_range)
                mean_votes = np.mean(sim.votes,axis=0)
                # print(mean_votes)
                results.append(np.abs(mean_votes[0] - mean_votes[1]))
            run_values.append(results)
            if r == 10:
                ax.plot(np.arange(0, 11), results, c="blue", alpha=0.1)
            if r == 20:
                ax.plot(np.arange(0, 11), results, c="orange", alpha=0.1)

        # Save data to numpy files
        np.save(f"results/voters_{num_voters}_non_voters_{num_non_voters}_r_{r_value}_iteration_analysis_plot", np.mean(run_values,axis=0))
        np.save(f"results/voters_{num_voters}_non_voters_{num_non_voters}_r_{r_value}_iteration_analysis_all_data", run_values)

        r_data.append(np.mean(run_values,axis=0))
        print(np.array(r_data).shape)

    for r in r_data:
        print(np.array(r).shape)
        print(r)
        ax.plot(np.arange(0, 11), r.squeeze())

    # ax.legend(["r = 10", "r = 20", "r = 30", "r = 40", "r = 50"])
    ax.grid()
    plt.show()




def run_animation(r: float) -> None:
    """
    Run an animation of the voting process for a given radius r.

    Parameters:
    r: float - the radius within which to look for neighbours
    """

    sim = Sim(50, 40)

    fig, ax, scat = plot_voting(sim, False)
    fig.patch.set_facecolor('#D2B48C')
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
                    c="gray", alpha=0.2, zorder=0)

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

def calculate_baseline(num_iterations_per_r, num_r_values):
        sim = Sim(100, 100)
        total_results = []
        all_data = []
        for r_value in range(0, 100, 100//num_r_values):
            print(f"---\n{r_value}---\n")
            results = []
            sim.reset(100, 100)
            for _ in range(num_iterations_per_r):
                r = r_value
                sim.reset(100, 100)
                points_within_range = voting.simulate_voting(sim, r, 0.5)
                real_majority = voting.get_majority(sim)
                for i in range(len(sim.votes)):
                    votes_neighbours = [vote for vote in sim.votes[points_within_range[i]] if vote[0] == 1 or vote[1] == 1]
                    # votes_neighbours.append(sim.votes[i])
                if len(votes_neighbours) == 0:
                    results.append(0)
                    continue
                    # print(votes_neighbours)
                tally = np.sum(votes_neighbours, axis=0)
                    # print(tally)
                if tally[0] > tally[1]:
                    local_vote = np.array([1, 0])
                elif tally[0] < tally[1]:
                    local_vote = np.array([0, 1])
                else:
                    local_vote = np.array([0, 0])
                if np.array_equal(local_vote, real_majority):
                    results.append(1)
                else:
                    results.append(0)


            total_results.append(np.mean(results))
            all_data.append(results)

        # Check that the number of results is correct
        assert len(total_results) == num_r_values

        # Save data to numpy files
        np.save(f"results/baseline_notself_{num_r_values}", total_results)
        np.save(f"results/baseline_notself_all_data_{num_r_values}", all_data)
        
        _, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
        ax.plot(np.arange(0, 100, 100//num_r_values), total_results)
        plt.show()


def randomize_positions_test(num_r_values: int, num_voters: int, num_non_voters: int, num_iterations_per_r: int) -> None:
    """
    Analyze the effect of the radius r on the voting outcome when positions are randomized.

    Parameters:
    num_r_values: int - the number of r values to analyze
    num_voters: int - the number of voters
    num_non_voters: int - the number of non-voters
    num_iterations_per_r: int - the number of iterations per r

    """
    _, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    for iterations in [0, 1, 2, 4]:
        total_results = []
        all_data = []

        total_cows = num_voters + num_non_voters
        sim = Sim(total_cows, num_voters)

        field_radius = 50*np.sqrt(total_cows/100)

        for r_value in range(0, 100, 100//num_r_values):
            print(f"---\n{r_value}---\n")
            results = []
            sim.reset(total_cows, num_voters, field_radius)
            for _ in range(num_iterations_per_r):
                r = r_value
                sim.reset(total_cows, num_voters, field_radius)
                points_within_range = voting.simulate_voting(sim, r, 0.5)
                # print(voting.get_majority(sim))
                voting.unvoted_vote(sim,points_within_range)
                for i in range(iterations):
                    #randomize
                    sim.randomize_positions()
                    points_within_range = sim.compute_neighbours(r)
                    voting.choose_local_majority(sim, points_within_range)
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

        ax.plot(np.arange(0, 100, 100//num_r_values), total_results)

    ax.legend(["0 iterations", "1 iteration", "2 iterations", "4 iterations"])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run analysis functions.")

    parser.add_argument("--run_animation", type=int, help="Run animation with given number of frames")
    parser.add_argument("--r_analysis", nargs=4, type=int, help="Run vision range analysis", metavar=("a", "b", "c", "d"))
    parser.add_argument("--histogram_analysis", type=int, help="Run histogram analysis with given number of bins")
    parser.add_argument("--influence_analysis", type=int, help="Run influence analysis with given steps")
    parser.add_argument("--uninformed_analysis", action="store_true", help="Run uninformed analysis")

    args = parser.parse_args()

    if any(vars(args).values()):
        if args.run_animation is not None:
            run_animation(args.run_animation)
        if args.r_analysis is not None:
            r_analysis(*args.r_analysis)
        if args.histogram_analysis is not None:
            histogram_analysis(args.histogram_analysis)
        if args.influence_analysis is not None:
            influence_analysis(args.influence_analysis)
        if args.uninformed_analysis:
            uninformed_analysis()
    else:
        print("No arguments provided. Use --help for options.")


if __name__ == "__main__":
    # main()
    run_animation(18)
    # iteration_analysis(100, 100)
    # r_analysis(50, 100, 0, 250)
    # calculate_baseline(1000, 25)
