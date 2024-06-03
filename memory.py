import random

from collections import deque

from params import MEMORY_CAPACITY
from transition import Transition


class Memory:
    def __init__(
            self,
            capacity=MEMORY_CAPACITY,
    ):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
