import gym

BATCH_SIZE = 128
# LR = 0.01
LR = 1e-4
GAMMA = 0.90
START_EPSILON = 0.1
DECAY = 0.9999
MEMORY_CAPACITY = 20000  # NB: could be increased
SAMPLE_FROM = 1000
Q_NETWORK_ITERATION = 100

# todo: deprecated, change to v1(?)
# https://github.com/openai/gym/wiki/CartPole-v0
env = gym.make("CartPole-v0", render_mode="rgb_array")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

DEVICE = 'mps'
