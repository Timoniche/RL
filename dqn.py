import torch
import numpy as np

from memory import Memory
from net import Net
from params import MEMORY_CAPACITY, BATCH_SIZE, Q_NETWORK_ITERATION, DEVICE
from transition import Transition


class DQN:
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net = Net()
        self.target_net = Net()  # NB: We need a target_net for stable evaluation.

        self.eval_net.to(DEVICE)
        self.target_net.to(DEVICE).eval()

        self.learn_step_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)

        # Define the self.optimizer and self.loss_func:

    def ready_to_learn(self):
        return self.memory.ready_to_sample()

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

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def _sample_batch(self):
        list_of_transitions = self.memory.sample(batch_size=BATCH_SIZE)
        states, actions, rewards, next_states, dones = Transition(*zip(*list_of_transitions))

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)

        return states, actions, rewards, next_states, dones

    def learn(self):
        states, actions, rewards, next_states, dones = self._sample_batch()

        # code for the q-learning update
        q_predict_all_actions = self.eval_net(states)
        q_predict = q_predict_all_actions[np.arange(BATCH_SIZE), actions]

        with torch.no_grad():
            q_next_all_actions = self.target_net(next_states)
            q_next = q_next_all_actions.max(dim=1).values
            for i in range(len(dones)):
                if dones[i]:
                    q_next[i] = 0.0

        # updating the target network parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
