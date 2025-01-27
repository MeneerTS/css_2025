import numpy as np
from herd import Herd

class Sim:
    """
    A class to keep track of bisons and their votes to simulate the voting process.

    Attributes:
    herd_size: int - the number of bisons in the herd
    herd: Herd - the herd of bisons
    locations: np.array - the locations of the bisons
    votes: np.array - the votes of the bisons
    num_voted: int - the number of bisons that have voted
    num_voters: int - the number of bisons that can vote

    Methods:
    reset: int, int -> None - reset the simulation to a new configuration
    compute_neighbours: float -> list - compute the neighbours of each point within a radius r
    """
    def __init__(self, herd_size=100, num_voters=50) -> None:
        assert num_voters <= herd_size
        self.herd_size = herd_size
        self.herd = Herd(self.herd_size, [100, 100], 50)
        # This should create a herd of herd_size bisons
        assert len(self.herd.bisons) == self.herd_size

        self.locations, self.votes = self.herd.as_numpy()
        # Every bison should have a location and a vote
        assert len(self.locations) == self.herd_size and len(self.votes) == self.herd_size

        self.num_voted = 0
        # The number of voters should start at 0
        assert self.num_voted == 0

        self.num_voters = num_voters
        # Only a total of herd_size bisons can vote
        assert self.num_voters <= self.herd_size


    def reset(self, herd_size=100, num_voters=50) -> None:
        self.herd.reset(herd_size)
        # This should reset to a herd of herd_size bisons
        assert len(self.herd.bisons) == self.herd_size

        self.locations, self.votes = self.herd.as_numpy()
        # Every bison should have a location and a vote
        assert len(self.locations) == self.herd_size and len(self.votes) == self.herd_size

        self.num_voted = 0
        # The number of voters should start at 0
        assert self.num_voted == 0

        self.num_voters = num_voters
        # Only a total of herd_size bisons can vote
        assert self.num_voters <= self.herd_size


    def compute_neighbours(self, r: float) -> list:
        """
        Compute the neighbours of each point within a radius r.

        Parameters:
        r: float - the radius within which to look for neighbours

        Returns:
        points_within_range: list - a list of neighbours for each point
        """

        # Compute pairwise distances
        pairwise_distances = np.linalg.norm(self.locations[:, np.newaxis] - self.locations, axis=2)

        # Find points within the range for each point
        points_within_range = [np.where((pairwise_distances[i] <= r) & (pairwise_distances[i] != 0))[0] for i in range(len(self.locations))]

        # Check that the number of points is the same as the number of locations
        assert len(points_within_range) == len(self.locations)

        return points_within_range