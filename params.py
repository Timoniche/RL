import gym

BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILON = 0.9
MEMORY_CAPACITY = 20000  # NB: could be increased
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

DEVICE = 'mps'
