import numpy as np
import matplotlib.pyplot as plt
from grid_world_environment import GridWorldEnvironment
from IPython.display import clear_output
from time import sleep
from tqdm import tqdm


class MontecarloAgent:
    def __init__(self,
                 env: GridWorldEnvironment,
                 discount_factor: float = 1,
                 epsilon: float = 0.5,
                 min_epsilon: float = 0.1,
                 q_discount_factor: float = 0.5,
                 max_iter: int = 10000):

        self.env = env
        self.discount_factor = discount_factor
        self.actions = self.env.actions    # Stay, up, down, left, right
        self.epsilon = epsilon    # Initial value for epsilon
        self.min_epsilon = min_epsilon    # Minimum value for epsilon
        self.q_discount_factor = q_discount_factor
        self.max_iter = max_iter

        # No need for a policy table, since actions are taken greedily from Q.
        self.q_value_table: np.ndarray = self.initialize_q_values()
        self.returns: dict[(int, int, int), list[int]] = dict()    # Stores the returns of each state-action pair
        self.returns_count: dict[(int, int, int), int] = dict()    # Counts the occurrence of each state-action pair
        self.cumulative_returns = list()    # Store cumulative returns of each episode

    def initialize_q_values(self) -> np.ndarray:
        matrix_dimension: (int, int, int) = self.env.size + (len(self.actions), )
        return np.full(matrix_dimension, 0.0, dtype=float)

    def get_greedy_action(self, state: (int, int)) -> (int, int):
        return self.actions[self.q_value_table[state].argmax()]

    def get_epsilon_greedy_action(self, state: (int, int)) -> (int, int):
        if np.random.rand() >= self.epsilon:
            return self.get_greedy_action(state)
        else:
            action_index: int = np.random.randint(0, len(self.actions))
            return self.actions[action_index]

    def generate_episode(self) -> list[tuple[any, any, any]]:
        trajectory = list()
        state: (int, int) = self.env.reset()    # Reset to initial position
        k = 0
        while True:
            k += 1
            action: (int, int) = self.get_epsilon_greedy_action(state)
            next_state, reward = self.env.step(action)
            trajectory.append((state, action, reward))
            if next_state == self.env.target or k == self.max_iter:
                break
            state = next_state
        return trajectory

    def learn(self, episode):
        """
        Update Q-values based on the episode using a first-visit incremental approach.
        :param episode: A list of (state, action, reward) tuples
        """
        visited_state_actions = set()  # To check if the state-action pair has been visited
        rewards = [x[2] for x in episode]
        G = 0  # Total return
        # Process the episode in reverse to calculate returns efficiently
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            action_idx = self.actions.index(action)
            state_action = state + (action_idx,)

            # Check for first visit to the state-action pair in the episode
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)
                G = self.discount_factor * G + reward  # Update return

                # Incremental update of Q-values
                # N(S, A): self.returns_count[state_action]
                # Q(S, A): self.q_value_table[state_action]
                # New estimate: Q(S, A) + (1 / N(S, A)) * (G - Q(S, A))

                # Update counts for first-visit
                if state_action not in self.returns:
                    self.returns[state_action] = [G]  # Initialize if first visit in all episodes
                    self.returns_count[state_action] = 1
                else:
                    self.returns_count[state_action] += 1  # Increment count
                    self.returns[state_action].append(G)  # Keep for potential analysis

                # Incremental update formula
                alpha = 1.0 / self.returns_count[state_action]
                self.q_value_table[state_action] += alpha * (G - self.q_value_table[state_action])

    @staticmethod
    def geom_alpha(x: float, k: int) -> float:
        if k == 0:
            return 1
        if x == 1.0:
            return 1 / k
        else:
            return (1 - x) / (x - x ** (k + 1))

    # TODO
    def _learn(self, episode: list) -> None:
        """
        Update Q-values based on the episode using a first-visit incremental approach.
        :param episode: A list of (state, action, reward) tuples
        """
        visited_state_actions = set()
        # rewards: list[int] = [x[2] for x in episode]
        g = 0    # Total return
        # Process the episode in reverse to calculate returns efficiently
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            action_index: int = self.actions.index(action)
            state_action: (int, int, int) = state + (action_index, )

            # Update return
            g = self.discount_factor * g + reward

            # Check for first visit to the state-action pair in the episode
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)
                '''# Update return
                g = self.discount_factor * g + reward'''

                if state_action not in self.returns_count:
                    #self.returns[state_action] = [g]
                    self.returns_count[state_action] = 1
                else:
                    #self.returns[state_action].append(g)
                    self.returns_count[state_action] += 1

                # Incremental update formula
                k = self.returns_count[state_action] - 1
                x = self.q_discount_factor
                self.q_value_table[state_action] = x * (
                        (self.q_value_table[state_action] * MontecarloAgent.geom_alpha(x, k + 1)
                         / MontecarloAgent.geom_alpha(x, k)) +
                        (MontecarloAgent.geom_alpha(x, k + 1) * g)
                )

    def train(self, n_episodes: int, plot: bool = False, plot_frequency: int = 50) -> None:
        for episode in tqdm(range(1, n_episodes + 1)):
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

    def animate_robot_movement(self, delay=0.5):
        # Continue until the target is reached
        while self.env.position != self.env.target:
            # Display the current state of the environment
            self.env.display()

            # Get the best action for the current state
            action = self.get_epsilon_greedy_action(self.env.position)

            # Take a step in the environment
            next_state, reward = self.env.step(action)

            # Wait for a moment
            sleep(delay)

            # Clear the output
            clear_output(wait=True)

        # Display the final state
        self.env.display()
        print("Target reached!")
