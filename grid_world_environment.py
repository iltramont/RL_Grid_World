import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pdb
import time
from IPython.display import clear_output

class ValidStateException(Exception):
  pass


class GirdWorldEnviroment:
    def __init__(self, size: (int, int), obstacles: tuple[(int, int)], position: (int, int), target: (int, int),
                 actions: tuple[(int, int)] = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
                 rewards: tuple[int] = (0, -5, -1),
                 malfunction_probability: float = 0.2):
        """
        Defines a grid-world environment.
        :param malfunction_probability:
        :param size: size of the environment size.
        :param obstacles: cells to be defined as obstacles.
        :param position: starting position.
        :param target: target position.
        :param actions: possible actions to be taken by the robot.
        :param rewards: actions' rewards.
        """
        self.size = size
        self.obstacles = obstacles  # for plotting purposes
        self.states = [(x, y) for x in range(size[0]) for y in range(size[1]) if (x, y) not in obstacles]
        self.target = target
        self.actions = actions
        self.rewards = rewards
        self.malfunction_probability = malfunction_probability

        if self.is_valid_state(position):
            self.initial_position = position  # for reset
            self.position = position
        else:
            raise ValidStateException

    def is_valid_state(self, state: (int, int)) -> bool:
        """
        Used to verify if a state is an obstacle or is out of bound.
        """
        return state in self.states

    def get_reward(self, current_state, action, next_state) -> int:
        if current_state == self.target:
            return self.rewards[0]
        elif current_state == next_state and action != (0, 0):
            return self.rewards[1]
        else:
            return self.rewards[2]

    def malfunction_action(self, action_input: (int, int)) -> (int, int):
        if action_input == (0, 0):
            error: int = (np.random.randint(1, 3) * 2) - 3
            x_mal: bool = np.random.random() < 0.5
            if x_mal:
                action_output: (int, int) = (error, 0)
            else:
                action_output: (int, int) = (0, error)
        ...

    # this should be modified if we introduce malfunctioning actuator
    def get_next_state(self, current_state, action):
        malfunction: bool = np.random.random() <= self.malfunction_probability
        if malfunction:
            clockwise: bool = np.random.random() < 0.5

        possible_next_state = (current_state[0] + action[0], current_state[1] + action[1])
        if self.is_valid_state(possible_next_state) and current_state != self.target:
            next_state = possible_next_state
        else:
            next_state = current_state
        return next_state

    def step(self, action):
        current_state = self.position
        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(current_state, action, next_state)
        self.position = next_state
        return next_state, reward

    def reset(self):
        self.position = self.initial_position
        return self.position

    def display(self):
        # Create a grid initialized to white (empty cells)
        grid = np.ones((self.size[0], self.size[1], 3))

        # Set obstacles to black
        for obstacle in self.obstacles:
            grid[obstacle] = [0, 0, 0]  # Black color

        # Set target to green
        grid[self.target] = [0, 1, 0]  # Green color

        # Set agent's position to blue
        grid[self.position] = [0, 0, 1]  # Blue color

        # Display the grid
        plt.imshow(grid)
        plt.xticks(range(self.size[1]))
        plt.yticks(range(self.size[0]))
        # Shift grid lines to the left and up by 0.5
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, self.size[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.size[0], 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        plt.show()