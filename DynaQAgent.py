import numpy as np


class DynaQAgent:
    def __init__(self, env, epsilon, gamma, lr, loop):
        self.env = env
        self.steps = loop
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr_rate = lr
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.transitions = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.uint8)
        self.rewards = np.zeros((env.observation_space.n, env.action_space.n))

    def sample(self, env):
        # Random state
        if all(np.sum(self.transitions, axis=1)) <= 0:
            state = np.random.randint(env.observation_space.n)
        else:
            state = np.random.choice(np.where(np.sum(self.transitions, axis=1) > 0)[0])
        # Random action
        if all(self.transitions[state]) <= 0:
            action = np.random.randint(env.action_space.n)
        else:
            action = np.random.choice(np.where(self.transitions[state] > 0)[0])
        return state, action

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])

    def update(self, state, state2, reward, action):
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = (1 - self.lr_rate) * self.Q[state, action] + self.lr_rate * target
        self.transitions[state, action] = state2
        self.rewards[state, action] = reward
        for i in range(self.steps):
            state, action = self.sample(self.env)
            state2 = self.transitions[state, action]
            reward = self.rewards[state, action]
            target = reward+self.gamma*np.max(self.Q[state2, :])
            self.Q[state, action] = (1 - self.lr_rate) * self.Q[state, action] + self.lr_rate * target
