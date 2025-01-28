import random
import numpy as np
from herd import Herd
from sim import Sim

def random_vote(herd: Herd, count: int, p = 0.5) -> None:
    """
    Select the first #count bisons and randomly assign a vote to them with probability p, and reset the vote of the remaining bisons.

    Parameters:
    herd: Herd - the herd of bisons
    count: int - the number of bisons to vote
    p: float - the probability of voting for [1, 0]
    """
    for i, bison in enumerate(herd.bisons):
        if i < count:
            if random.random() < p:
                bison.vote = np.array([1, 0])
            else:
                bison.vote = np.array([0, 1])
        else:
            # Reset the vote of the remaining bisons
            bison.vote = np.array([0, 0])

def random_vote_indexed(herd: Herd, i: int, p = 0.5) -> None:
    """
    Randomly assign a vote to the i-th bison with probability p.

    Parameters:
    herd: Herd - the herd of bisons
    i: int - the index of the bison to vote
    p: float - the probability of voting for [1, 0]
    """

    if random.random() < p:
        herd.bisons[i].vote = np.array([1, 0])
    else:
        herd.bisons[i].vote = np.array([0, 1])

def simulate_voting(sim: Sim, r: float) -> list:
    """
    Simulate a voting cycle

    Parameters:
    sim: Sim - the simulation object
    r: float - the radius within which to look for neighbours

    Returns:
    points_within_range: list - a list of neighbours for each point for future use
    """
    points_within_range = sim.compute_neighbours(r)
    while True:
        if sim.num_voted >= sim.num_voters:
            return points_within_range
        if sim.num_voted < sim.num_voters:

            # Tally all local votes
            votes_neighbours = [vote for vote in sim.votes[points_within_range[sim.num_voted]] if vote[0] == 1 or vote[1] == 1]
            # Compute the influence of the neighbours
            influence = np.mean(votes_neighbours, axis=0) if len(votes_neighbours) > 0 else np.array([0, 0])
            influence = (influence[0] - influence[1]) / 2

            # Influence should be between -0.5 and 0.5
            assert -0.5 <= influence <= 0.5

            random_vote_indexed(sim.herd, sim.num_voted, 0.5 + influence)

            # Update variables
            sim.locations, sim.votes = sim.herd.as_numpy()
            sim.num_voted += 1


def unvoted_vote(sim: Sim, points_within_range: list, use_random=True) -> None:
    """
    Simulate the unvoted bisons voting based on the votes of their neighbours,
    or randomly if there are no neighbours when use_random is enabled, otherwise no vote is cast.

    Parameters:
    sim: Sim - the simulation object
    points_within_range: list - the list of neighbours for each point
    use_random: bool - whether to use random voting
    """
    for unvotedindex in range(sim.num_voted,sim.herd_size):
        votes_neighbours = [vote for vote in sim.votes[points_within_range[unvotedindex]] if vote[0] == 1 or vote[1] == 1]
        influence = np.mean(votes_neighbours, axis=0) if len(votes_neighbours) > 0 else np.array([0, 0])

        if len(votes_neighbours) > 0:
            # Check if there is a majority in our neighbours
            if influence[0] == influence[1]:
                # Do a random vote if our neighbours are divided
                random_vote_indexed(sim.herd, unvotedindex, 0.5)
            else:
                # Otherwise vote with the majority
                influence = influence[0] > influence[1]
                random_vote_indexed(sim.herd, unvotedindex, influence)
        elif use_random:
            # If there are no neighbours and use_random is enabled, do a random vote
            random_vote_indexed(sim.herd, unvotedindex, 0.5)

    # Update variables
    sim.locations, sim.votes = sim.herd.as_numpy()

def choose_local_majority(sim: Sim, points_within_range: list) -> None:
    """
    Do a round where every bison figures out where their local majority is going. They then change their vote to that.

    Parameters:
    sim: Sim - the simulation object
    points_within_range: list - the list of neighbours for each point    
    """

    for unvotedindex in range(0,sim.herd_size):
        votes_neighbours = [vote for vote in sim.votes[points_within_range[unvotedindex]] if vote[0] == 1 or vote[1] == 1]
        influence = np.mean(votes_neighbours, axis=0) if len(votes_neighbours) > 0 else np.array([0, 0])

        if len(votes_neighbours) > 0:
            # Check if there is a majority in our neighbours
            #If our neighbours are divided we stay with our original vote
            if influence[0] != influence[1]:
                influence = influence[0] > influence[1]
                random_vote_indexed(sim.herd, unvotedindex, influence)
        #If there are no neighbours we keep our vote

    # Update variables
    sim.locations, sim.votes = sim.herd.as_numpy()
