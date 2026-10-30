"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # TODO: build a 2D float64 numpy array of zeros sized by states and actions.
    return np.zeros((num_states, num_actions), dtype='float64')

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    # TODO: index the row for `state` and return its maximum value
    return np.max(q_table[state])

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    return int(np.argmax(q_table[state]))

# Step 4 - sample_random_action
def sample_random_action(action_space):
    """draw a uniformly random action from the given Gymnasium action space"""
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    u = rng.uniform(0,1)
    return u < epsilon

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # TODO: with prob epsilon explore via action_space, else pick a max-Q action (random among ties)
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    else:
        q_values = q_table[state]
        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(rng.choice(best_actions))

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    if done:
        return reward
    else:
        return reward + gamma * max_q_value(q_table, next_state)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    return target - q_table[state, action]

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)
    q_table[state, action] += alpha * error
    return q_table[state, action]

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)
    return (int(next_state), float(reward), done)

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):        
        next_state, reward, done = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        
        total_reward += reward
        state = next_state
        if done:
            break

    return total_reward

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.8, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99, seed=0, max_steps=200):
    rng = np.random.default_rng(seed)
    returns = []
    q_table = init_q_table(16,4)
    for episode in range(num_episodes):
        env.reset(seed=seed)
        action_space = env.action_space.seed(seed)

        reward = run_training_episode(env, q_table, epsilon_start, alpha, gamma, rng, max_steps)
        returns.append(reward)

        epsilon = decay_epsilon(epsilon_start, epsilon_decay, epsilon_min)
        epsilon_start = epsilon
    
    return (q_table, returns)

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

