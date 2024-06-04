from matplotlib import pyplot as plt


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('CartPole-v0')
    plt.show()


rewards_filename = 'rewards_2024_06_05_02_01.txt'


def main():
    with open(rewards_filename, 'r') as f:
        line = f.read()
        rewards = list(map(lambda x: float(x), line.split()))

    plot_rewards(rewards)


if __name__ == '__main__':
    main()
