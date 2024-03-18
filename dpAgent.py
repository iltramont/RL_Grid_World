from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from IPython.display import clear_output
from numpy import ndarray, dtype
from grid_world_environment import GridWorldEnvironment


class DPAgent:
    """
    Dynamic programming agent. Must be used only when there isn't malfunction probability.
    """
    def __init__(self, env: GridWorldEnvironment, discount_factor=1):

        self.env = env
        self.discount_factor = discount_factor
        self.actions = self.env.actions    # stay, up, down, right, left

        self.value_table = self.initialize_values()
        self.policy_table = self.initialize_policy()

    def initialize_values(self) -> ndarray[Any, dtype[Any]]:
        # Initialize value function for all states
        values = np.full(self.env.size, np.nan)
        # Set value to 0 for admissible states
        for state in self.env.states:
            values[state] = 0
        return values

    def initialize_policy(self) -> ndarray[Any, dtype[Any]]:
        # Initialize policy table for all states
        policy = np.full(self.env.size + (len(self.actions),), np.nan)
        # Set uniform policy for admissible states
        for state in self.env.states:
            policy[state] = np.ones(len(self.actions)) / len(self.actions)
        return policy

    def get_optimal_action(self, state) -> (int, int):
        return self.actions[np.argmax(self.policy_table[state])]

    def policy_eval(self, theta=0.001):
        while True:
            delta = 0
            for state in self.env.states:
                v = 0
                for action in self.actions:
                    next_state = self.env.get_next_state(state, action)
                    reward = self.env.get_reward(state, action, next_state)
                    v += self.policy_table[state + (self.actions.index(action),)] * (
                                reward + self.discount_factor * self.value_table[next_state])
                delta = max(delta, np.abs(self.value_table[state] - v))
                self.value_table[state] = v

            if delta < theta:
                break

    def policy_improvement(self) -> bool:
        policy_stable = True
        for state in self.env.states:
            # Handle terminal state if necessary
            if state == self.env.target:
                continue

            chosen_action = self.get_optimal_action(state)
            action_values = []
            for action in self.actions:
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(state, action, next_state)
                action_values.append(reward + self.discount_factor * self.value_table[next_state])
            best_action = self.actions[np.argmax(action_values)]

            if chosen_action != best_action:
                policy_stable = False

            self.policy_table[state] = np.eye(5)[self.actions.index(best_action)]
        return policy_stable

    def policy_iteration(self, theta: float):
        self.initialize_values()
        self.initialize_policy()
        cont = 1
        while True:
            print(f'Iteration {cont}...')
            self.policy_eval(theta)
            if self.policy_improvement():
                break
            cont += 1
        return self.policy_table

    def plot_value_table(self):
        # Plot Optimal Value Function and Optimal Policy
        value_table_processed = np.nan_to_num(self.value_table, nan=5 * min(self.value_table[0]))
        plt.imshow(value_table_processed, cmap='hot', interpolation='nearest', vmin=2 * min(self.value_table[0]))
        plt.colorbar()
        plt.title('Heatmap of Gridworld State Values')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')

        # Define the axis for plotting
        ax = plt.gca()

        # Plot the policy
        for i in range(self.env.size[0]):
            for j in range(self.env.size[1]):
                if (i, j) in self.env.obstacles:
                    continue
                best_action = self.get_optimal_action((i, j))
                if best_action == (0, 0):
                    # Stay action, plot a circle
                    circle = plt.Circle((j, i), 0.1, color='blue', zorder=5)  # Adjusted y coordinate
                    ax.add_patch(circle)
                else:
                    # Plot an arrow for the action
                    dx, dy = best_action[1], -best_action[0]  # Adjust direction for matrix coordinate system
                    # Flip the dy value to correct the orientation
                    plt.arrow(j, i, dx * 0.3, -dy * 0.3, color='blue', head_width=0.1, head_length=0.1,
                              zorder=5)  # Adjusted y coordinate and dy
        plt.show()

    def animate_robot_movement(self, delay=0.5):
        # Continue until the target is reached
        while self.env.position != self.env.target:
            # Display the current state of the environment
            self.env.display()

            # Get the best action for the current state
            action = self.get_optimal_action(self.env.position)

            # Take a step in the environment
            next_state, reward = self.env.step(action)

            # Wait for a moment
            sleep(delay)

            # Clear the output
            clear_output(wait=True)

        # Display the final state
        self.env.display()
        print("Target reached!")
