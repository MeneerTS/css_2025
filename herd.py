import numpy as np

class Bison:
    """
    A Bison is an animal that can vote.

    Attributes:
    position: np.array - the position of the bison
    vote: np.array - the vote of the bison
    """
    def __init__(self, position: np.array, vote: np.array) -> None:
        """
        Initialize a Bison.

        Parameters:
        position: np.array - the position of the bison
        vote: np.array - the vote of the bison
        """

        self.position = position
        self.vote = vote

class Herd:
    """
    A Herd is a collection of bisons.

    Attributes:
    bisons: list - a list of bisons

    Methods:
    add_bison: Bison -> None - add a bison to the herd
    as_numpy: None -> np.array - return the locations and votes of the bisons in the herd
    reset: int, list, float -> None - reset the herd to a new configuration
    """

    def __init__(self, count = 100, center = [100, 100], radius = 50) -> None:
        """
        Initialize a herd of bisons.

        Parameters:
        count: int - the number of bisons in the herd
        center: list - the center of the herd
        radius: float - the radius of the herd
        """
        self.reset(count, center, radius)


    def add_bison(self, bison: Bison) -> None:
        """
        Add a bison to the herd.

        Parameters:
        bison: Bison - the bison to add to the herd
        """
        self.bisons.append(bison)


    def as_numpy(self) -> tuple:
        """Return the locations and votes of the bisons in the herd as 2 numpy arrays."""
        locations = []
        votes = []
        for bison in self.bisons:
            locations.append(list(bison.position))
            votes.append(list(bison.vote))
        return np.array(locations), np.array(votes)

    def randomize_positions(self,center,radius) -> None:
        """
        Randomizes all positions while keeping votes etc the same
        center: list - the center of the herd
        radius: float - the radius of the herd
        """
        #The count stays the same
        newpositions = self.generate_positions(len(self.bisons),center,radius)
        
        for i, bison in enumerate(self.bisons):
            bison.position = newpositions[i]

    def generate_positions(self, count,center,radius) -> list:
        """
        Generates a count sized list of (x,y) positions uniformly distributed in 
        a circle of radius radius centered on center

        count: int - the number of positions
        center: list - the center of the circle
        radius: float - the radius of the circle
        """

        for _ in range(count):
            r = radius * np.sqrt(np.random.rand(count))
            theta = 2 * np.pi * np.random.rand(count)
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            positions = np.stack((x, y), axis=-1)
        return positions
    def reset(self, count = 100, center = [100, 100], radius = 50) -> None:
        """
        Resets the herd of bisons to a new configuration.

        Parameters:
        count: int - the number of bisons in the herd
        center: list - the center of the herd
        radius: float - the radius of the herd
        """
        self.bisons = []
        positions = self.generate_positions(count,center,radius)
        self.bisons = [Bison(pos, np.zeros(2)) for pos in positions]

        # The number of bisons should be equal to count
        assert len(self.bisons) == count

