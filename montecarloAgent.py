import numpy as np
import matplotlib.pyplot as plt
from grid_world_environment import GridWorldEnvironment
from IPython.display import clear_output


class MontecarloAgent:
    def __init__(self,
                 env: GridWorldEnvironment,
                 discount_factor: float = 1,
                 epsilon: float = 0.5,
                 min_epsilon: float = 0.1):

        self.env = env
        self.discount_factor = discount_factor
        self.actions = self.env.actions    # Stay, up, down, left, right
        self.epsilon = epsilon    # Initial value for epsilon
        self.min_epsilon = min_epsilon    # Minimum value for epsilon

        # No need for a policy table, since actions are taken greedily from Q.
        self.q_value_table: np.ndarray = self.initialize_q_values()
        self.returns: dict[(int, int, int), list[int]] = dict()    # Stores the returns of each state-action pair
        self.returns_count: dict[(int, int, int), int] = dict()    # Counts the occurrence of each state-action pair
        self.cumulative_returns = list()    # Store cumulative returns of each episode

    def initialize_q_values(self) -> np.ndarray:
        matrix_dimension: (int, int, int) = self.env.size + (len(self.actions), )
        return np.full(matrix_dimension, 0.0, dtype=float)

    def get_greedy_action(self, state: (int, int)) -> (int, int):
        action_index: int = np.argmax(self.q_value_table[state])
        return self.actions[action_index]

    def get_epsilon_greedy_action(self, state: (int, int)) -> (int, int):
        if np.random.rand() >= self.epsilon:
            return self.get_greedy_action(state)
        else:
            action_index: int = np.random.randint(0, len(self.actions))
            return self.actions[action_index]

    def generate_episode(self) -> list[(int, int), (int, int), int]:
        trajectory = list()
        state: (int, int) = self.env.reset()    # Reset to initial position
        while True:
            action: (int, int) = self.get_epsilon_greedy_action(state)
            next_state, reward = self.env.step(action)
            trajectory.append((state, action, reward))
            if next_state == self.env.target:
                break
            state = next_state
        return trajectory

    def learn(self, episode: list) -> None:
        """
        Update Q-values based on the episode using a first-visit incremental approach.
        :param episode: A list of (state, action, reward) tuples
        """
        visited_state_actions = set()
        rewards: list[int] = [x[2] for x in episode]
        g = 0    # Total return
        # Process the episode in reverse to calculate returns efficiently
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            action_index: int = self.actions.index(action)
            state_action: (int, int, int) = state + (action_index, )

            # Check for first visit to the state-action pair in the episode
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)
                # Update return
                g = self.discount_factor * g + reward

                if state_action not in self.returns:
                    self.returns[state_action] = [g]
                    self.returns_count[state_action] = 1
                else:
                    self.returns[state_action].append(g)
                    self.returns_count[state_action] += 1

                # Incremental update formula
                self.q_value_table[state_action] += \
                    (g - self.q_value_table[state_action]) / self.returns_count[state_action]

    def train(self, n_episodes: int, plot: bool = False, plot_frequency: int = 50) -> None:
        for episode in range(1, n_episodes + 1):
            self.epsilon = max(self.min_epsilon, 1 / episode)

            trajectory = self.generate_episode()
            self.learn(trajectory)

            self.cumulative_returns.append(sum([t[2] for t in trajectory]))
            self.update_plot(episode, plot, plot_frequency)

    def get_value_table(self) -> np.ndarray:
        return np.max(self.q_value_table, axis=2)

    def update_plot(self, episode, plot: bool, plot_frequency: int = 50):
        if plot and episode % plot_frequency == 0:
            # Clear output
            clear_output(wait=True)
            plt.figure(figsize=(21, 9))
            plt.plot(self.cumulative_returns, label="Cumulative returns")
            plt.xlabel('Episodes', fontsize="large")
            plt.ylabel('Cumulative Return', fontsize="large")
            plt.title('Cumulative Return per Episode', fontsize="xx-large")
            plt.legend()
            plt.show()
            # Clear the figure to prevent overlapping of plots
            plt.close()

    def plot_value_table(self):
        value_table = self.get_value_table()
        # Plot Optimal Value Function and Optimal Policy
        value_table_processed = np.nan_to_num(value_table, nan=5 * min(value_table[0]))
        plt.imshow(value_table_processed, cmap='hot', interpolation='nearest', vmin=2 * min(value_table[0]))
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
                best_action = self.get_greedy_action((i, j))
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
