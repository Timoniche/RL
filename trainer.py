from DQN import DQN
from params import env, MEMORY_CAPACITY


def train():
    dqn = DQN()
    episodes = 400
    print("Need to collect (actions, states, rewards, next_statex)....")
    for i in range(episodes):
        # state: ndarray: (4,)
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state[0])
            next_state, reward, done, info = env.step(action)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state


def main():
    train()


if __name__ == '__main__':
    main()
