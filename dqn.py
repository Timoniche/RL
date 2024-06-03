import torch
import numpy as np

from net import Net
from params import MEMORY_CAPACITY, NUM_STATES, BATCH_SIZE, Q_NETWORK_ITERATION, DEVICE


class DQN():
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net = Net()
        self.target_net = Net()  # We need a target_net for stable evaluation.

        self.eval_net.to(DEVICE)
        self.target_net.to(DEVICE).eval()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))

        # Define the self.optimizer and self.loss_func:

    def choose_action(self, state):
        # todo: epsilon-strategy here

        # Write your code for an epsilon-greedy exploration.
        # With probability 1-EPISILON choose a random action.
        return self.act(state)

    @torch.no_grad()
    def act(
            self,
            state: np.ndarray
    ):
        self.eval_net.eval()

        tensor_state = torch.FloatTensor(state).to(DEVICE)  # 1D array (state of 4 params)
        action_q_values = self.eval_net(tensor_state)
        action = torch.argmax(action_q_values).item()  # 0 or 1

        self.eval_net.train()

        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update(self):
        # sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        # Write your code for the q-learning update

        # update the target network parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
