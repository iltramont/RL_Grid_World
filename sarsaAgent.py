import matplotlib.pyplot as plt
import numpy as np
from grid_world_environment import GridWorldEnvironment
from IPython.display import clear_output
from time import sleep


class SarsaAgent:
    def __init__(self, env: GridWorldEnvironment,
                 discount_factor: float = 1,
                 alpha: float = 0.5,
                 epsilon: float = 0.5,
                 min_alpha: float = 0.1,
                 min_epsilon: float = 0.1,
                 gamma_epsilon: float = 2/3,
                 gamma_alpha: float = 2/3):

        self.env = env
        self.discount_factor = discount_factor

        # Initial values for alpha and epsilon
        self.alpha = alpha
        self.epsilon = epsilon

        # Minimum values for alpha and epsilon
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon

        # Used to compute epsilon and alpha
        self.gamma_epsilon = gamma_epsilon
        self.gamma_alpha = gamma_alpha

        self.actions = self.env.actions  # stay, up, down, right, left
        self.q_value_table = self.initialize_q_values()
        # no need for a policy table, since the actions are taken greedily from Q
        # Initialize list for storing cumulative returns for plotting purposes
        self.cumulative_returns = []

    def initialize_q_values(self) -> np.ndarray:
        """
        Initialize q value function for all states-action pair
        """
        values = np.full(self.env.size + (len(self.actions),), 0.0, dtype=float)  # attention: should be float
        return values

    def get_epsilon_greedy_action(self, state) -> (int, int):
        if np.random.uniform(0, 1) >= self.epsilon:
            action = self.get_greedy_action(state)
        else:
            action_idx = np.random.choice(range(len(self.actions)))
            action = self.actions[action_idx]
        return action

    def get_greedy_action(self, state) -> (int, int):
        action_idx = np.argmax(self.q_value_table[state])
        return self.actions[action_idx]

    def get_optimal_action(self, state) -> (int, int):
        return self.get_greedy_action(state)

    def learn(self, state, action, reward, next_state, next_action) -> None:

        # prediction = Q(S,A)
        prediction: float = self.q_value_table[state + (self.actions.index(action), )]

        # target = R + gamma * Q(S',A')
        target: float = reward + self.discount_factor * self.q_value_table[next_state + (self.actions.index(next_action), )]

        td_error: float = target - prediction

        # update:  Q(S,A) = Q(S,A) + alpha * td_error
        self.q_value_table[state + (self.actions.index(action),)] += self.alpha * td_error

    def get_epsilon(self, episode: int) -> float:
        return max(self.min_epsilon, 1 / episode ** self.gamma_epsilon)

    def get_alpha(self, episode: int) -> float:
        return max(self.min_alpha, 1 / episode ** self.gamma_alpha)


    def train(self, episodes: int, plot: bool = True, plot_frequency: int = 50) -> None:

        for episode in range(1, episodes + 1):

            # Update alpha and epsilon with decay, but not less than their minimum values
            self.epsilon = self.get_epsilon(episode)
            self.alpha = self.get_alpha(episode)

            state = self.env.reset()  # Reset environment for a new episode
            action = self.get_epsilon_greedy_action(state)

            total_reward = 0  # Initialize the return for the episode to 0

            while True:
                next_state, reward = self.env.step(action)
                # SARSA
                next_action = self.get_epsilon_greedy_action(next_state)
                self.learn(state, action, reward, next_state, next_action)

                total_reward += reward    # Accumulate reward

                state = next_state
                action = next_action

                if state == self.env.target:
                    break

            # Store cumulative return for this episode
            self.cumulative_returns.append(total_reward)

            if plot and episode % plot_frequency == 0:
                # Clear the output
                clear_output(wait=True)

                # Update plot after each episode
                last_episodes_to_plot: int = min(500, episode)
                self.update_plot(last_episodes_to_plot)

    def get_value_table(self) -> np.ndarray:
        """
        Assuming q_value_table is a 3D numpy array with dimensions [x, y, action]
        """
        value_table = np.max(self.q_value_table, axis=2)
        return value_table

    def plot_value_table(self):
        # Plot Optimal Value Function and Optimal Policy
        value_table_processed = np.nan_to_num(self.get_value_table(), nan=5 * min(self.get_value_table()[0]))
        plt.imshow(value_table_processed, cmap='hot', interpolation='nearest', vmin=2 * min(self.get_value_table()[0]))
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

    def update_plot(self, last_episodes_to_plot: int):
        plt.figure(figsize=(21, 9))
        x: list[int] = [i for i in range(1, len(self.cumulative_returns) + 1)][-last_episodes_to_plot:]
        y: list[float] = self.cumulative_returns[-last_episodes_to_plot:]
        mean_return: float = np.mean(y)
        plt.plot(x, y, label='Cumulative Return', color="blue")
        plt.axhline(mean_return,
                    label=f'Mean Return last {last_episodes_to_plot} episodes = {mean_return:.2f}', color='red')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Return per Episode')
        plt.legend(loc='lower right')
        plt.show()
        # Clear the figure to prevent overlapping of plots
        plt.close()

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