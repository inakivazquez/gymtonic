import random
import time
import numpy as np

class QTable:
    def __init__(self,n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.table = np.zeros((n_states, n_actions))

    def __getitem__(self, key):
        return self.table[key]

    def __str__(self):
        return_string = "State\t"
        for a in range(self.n_actions):            
            return_string += "{:>9}".format("A" + str(a))
        return_string += "\n"
        for state in range(self.n_states):
            return_string += "S"  +str(state) + ":\t"
            for action in range(self.n_actions):
                return_string += "{:9.4f}".format(self.table[state][action])
            return_string += "\n"
        return return_string

class ReturnsTable:
    def __init__(self,n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.table = np.empty((n_states, n_actions), dtype=list)

    def __getitem__(self, key):
        return self.table[key]

def random_policy(q_values):
    return random.randint(0,len(q_values)-1)

def max_policy(q_values):
    max_actions_indexes = np.argwhere(q_values == np.max(q_values))[0]
    return random.choice(max_actions_indexes)

class EpsilonGreedyPolicy():
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0,len(q_values)-1)
        else:
            return max_policy(q_values)

def weighted_policy(q_values):
    if sum(q_values) == 0:
        return random_policy(q_values)
    else:
        return random.choices(range(len(q_values)), weights=q_values)[0]

def softmax_policy(q_values):
    e = np.exp(q_values)
    softmax_values = e / e.sum()
    return random.choices(range(len(q_values)), weights=softmax_values)[0]

def print_duration(func):
    #print("----- Start measuring time -----")
    start_time = time.time()
    func()
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))


def evaluate_policy(env, q_table, policy, n_episodes:int=100, max_total_steps = 1_000_000, verbose=False):
    total_steps = 0
    total_reward = 0

    for n in range(n_episodes):
        episode_reward = 0
        obs, _ = env.reset()
        terminated = truncated = False
        if verbose:
            print("--- EPISODE STARTS ---")
            env.render()

        while not terminated and not truncated and total_steps < max_total_steps:
            selected_action = policy(q_table[obs])
            obs, reward, terminated, truncated, _ = env.step(selected_action)
            episode_reward += reward

            total_steps += 1

            if verbose:
                env.render()

        total_reward += episode_reward
        if total_steps >= max_total_steps: # in the case the episode is infinite
            break

    if total_steps >= max_total_steps:
        print("Average reward per episode: no episode completed")
        return None
    else:
        avg_reward = total_reward/n_episodes
        avg_steps = total_steps/n_episodes
        print("Average reward per episode: {:.4f}".format(avg_reward))
        print("Average steps per episode: {:.4f}".format(avg_steps))
        return avg_reward, avg_steps

def evaluate_policy_by_steps(env, q_table, policy, n_steps:int=100, verbose=False):
    obs = env.reset()
    if verbose:
        env.render()
    selected_action = policy(q_table[obs])
    total_reward = 0
    episode_reward = 0
    num_episodes = 0

    for n in range(n_steps):
        obs, reward, done, _ = env.step(selected_action)
        episode_reward += reward

        if verbose:
            env.render()

        selected_action = policy(q_table[obs])

        if done:
            # reset the environment and reinitialize trajectory
            if verbose:
                print("--- EPISODE STARTS ---")
            obs = env.reset()
            selected_action = policy(q_table[obs])
            if verbose:
                env.render()
            total_reward += episode_reward
            num_episodes+=1
            episode_reward = 0
            last_episode_step = n

    if num_episodes != 0:
        avg_reward = total_reward/num_episodes
        avg_steps = (last_episode_step+1)/num_episodes
        print("Average reward per episode: {:.4f}".format(avg_reward))
        print("Average steps per episode: {:.4f}".format(avg_steps))
        return avg_reward, avg_steps
    else:
        print("Average reward per episode: no episode completed")
        return None

