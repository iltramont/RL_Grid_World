import numpy as np
import matplotlib.pyplot as plt
from grid_world_environment import GirdWorldEnvironment


class DPAgent:
    def __init__(self, env: GirdWorldEnvironment, discount_factor: float = 1):
        self.env = env
        self.discount_factor = discount_factor
        # self.actions = self.env.actions
        self.value_table: np.ndarray = self.initialize_values()
        self.policy_table: np.ndarray = self.initialize_policy()

    def initialize_values(self) -> np.ndarray:
        """
        Initialize state value with zeros
        """
        values: np.ndarray = np.full(self.env.size, np.nan)
        for state in self.env.states:
            values[state] = 0
        return values

    def initialize_policy(self):
        """
        Initialize policy with equal probability for each action
        """
        policy_dimensions: (int, int, int) = self.env.size + (len(self.env.actions),)
        policy: np.ndarray = np.full(policy_dimensions, np.nan)
        for state in self.env.states:
            # Initialize with equal probability for each action
            policy[state] = np.ones(len(self.env.actions)) / len(self.env.actions)
        return policy

    def get_optimal_action(self, state: (int, int)) -> (int, int):
        """
        Get action according to the policy. Returns the action with the highest probability.
        """
        return self.env.actions[np.argmax(self.policy_table[state])[0]]

    def policy_eval(self, theta=1) -> None:
        """
        Evaluate current policy
        """
        while True:
            delta = 0
            # Compute state value for each state
            for state in self.env.states:  # Swipe
                # Initialize state value
                v = 0
                # Sum for each possible action
                for action_index in range(len(self.env.actions)):
                    action: (int, int) = self.env.actions[action_index]
                    action_prob: float = self.policy_table[state][action_index]    # p(a|s)
                    u = 0
                    for state_prime in [(state[0] + action[0], state[1] + action[1]) for action in self.env.actions]:

                    next_state = self.env.get_next_state(state, action)
                    reward = self.env.get_reward(state, action, next_state)

                    v = v + action_prob * (reward + self.discount_factor * self.value_table[next_state])

                new_delta = np.abs(v - self.value_table[state])
                self.value_table[state] = v
                delta = max(delta, new_delta)
            # delta = max(delta, np.abs(v - self.value_table[state]))
            # self.value_table[state] = v
            # print(delta)
            if delta < theta:
                break

    def policy_improvement(self) -> bool:
        policy_stable = True
        for state in self.env.states:
            if state == self.env.target:
                continue
            chosen_action = self.get_optimal_action(state)
            action_values = []
            for action in self.env.actions:
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(state, action, next_state)
                action_values.append(reward + self.discount_factor * self.value_table[next_state])

            best_action = self.env.actions[np.argmax(action_values)]
            if best_action != chosen_action:
                policy_stable = False
            self.policy_table[state] = np.eye(len(self.env.actions))[self.env.actions.index(best_action)]
        return policy_stable

    def policy_iteration(self, theta=1) -> np.ndarray:
        self.value_table = self.initialize_values()
        self.policy_table = self.initialize_policy()
        counter = 1
        while True:
            print(f"Iteration {counter}...")
            self.policy_eval(theta=theta)
            policy_stable = self.policy_improvement()
            if policy_stable:
                # self.policy_eval(theta=theta)
                print(f"Optimal policy found.")
                break
            counter += 1
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