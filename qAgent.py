from sarsaAgent import SarsaAgent
from grid_world_environment import GridWorldEnvironment
from IPython.display import clear_output


class QLearningAgent(SarsaAgent):
    def __init__(self, env: GridWorldEnvironment,
                 discount_factor: float = 1,
                 alpha: float = 0.5,
                 epsilon: float = 0.5,
                 min_alpha: float = 0.1,
                 min_epsilon: float = 0.1,
                 gamma_epsilon: float = 2 / 3,
                 gamma_alpha: float = 2 / 3):
        super().__init__(env, discount_factor, alpha, epsilon, min_alpha, min_epsilon, gamma_epsilon, gamma_alpha)

    def learn(self, state, action, reward, next_state, next_action=None):

        # prediction = Q(S,A)
        prediction = self.q_value_table[state + (self.actions.index(action),)]

        # target = R + gamma * max_a Q(S',a)
        next_action_greedy = self.get_greedy_action(next_state)

        target = reward + self.discount_factor * self.q_value_table[
            next_state + (self.actions.index(next_action_greedy),)]

        td_error = target - prediction

        # update:  Q(S,A) = Q(S,A) + alpha * td_error
        self.q_value_table[state + (self.actions.index(action),)] = self.q_value_table[state + (
            self.actions.index(action),)] + self.alpha * td_error

    def train(self, episodes: int, plot: bool = True, plot_frequency: int = 50):

        for episode in range(1, episodes + 1):

            # Update alpha and epsilon with decay, but not less than their minimum values
            self.epsilon = self.get_epsilon(episode)
            self.alpha = self.get_alpha(episode)

            state = self.env.reset()  # Reset environment for a new episode
            total_reward = 0  # Initialize the return for the episode to 0

            while True:
                action = self.get_epsilon_greedy_action(state)
                next_state, reward = self.env.step(action)
                self.learn(state, action, reward, next_state)

                total_reward += reward  # Accumulate reward

                state = next_state

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

    def get_optimal_action(self, state):
        return self.get_greedy_action(state)