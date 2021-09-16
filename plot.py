import numpy as np
import matplotlib.pyplot as plt


def plot_rwd_by_epd(rewards, epd_range=300):
    sum_rewards = []
    index = []
    for i in range(0, len(rewards), epd_range):
        index.append(i)
        sum_rewards.append(sum(rewards[i:i + epd_range]))

    plt.plot(index, sum_rewards)
    plt.ylabel('sum reward')
    plt.xlabel('episodes')
    plt.show()


def plot_steps_by_epd(steps, epd_range=300):
    avg_steps = []
    index = []
    for i in range(0, len(steps), epd_range):
        index.append(i)
        avg_steps.append(np.mean(steps[i:i + epd_range]))
    plt.plot(index, avg_steps)
    plt.ylabel('steps')
    plt.xlabel('episodes')
    plt.show()
