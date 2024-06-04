import random

from collections import deque

from params import MEMORY_CAPACITY, SAMPLE_FROM
from transition import Transition


class Memory:
    def __init__(
            self,
            capacity=MEMORY_CAPACITY,
            sample_from=SAMPLE_FROM,
    ):
        self.sample_from = sample_from
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def ready_to_sample(self):
        return len(self) >= self.sample_from

    def __len__(self):
        return len(self.memory)
