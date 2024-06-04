from matplotlib import pyplot as plt


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('CartPole-v0 Rewards')
    plt.show()


def plot_losses(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('CartPole-v0 Losses')
    plt.show()


rewards_filename = 'metrics/rewards_2024_06_05_02_29.txt'
losses_filename = 'metrics/losses_2024_06_05_02_29.txt'


def read_floats(filename):
    with open(filename, 'r') as f:
        line = f.read()
        floats = list(map(lambda x: float(x), line.split()))

    return floats


def main():
    plot_rewards(read_floats(rewards_filename))
    plot_losses(read_floats(losses_filename))


if __name__ == '__main__':
    main()
