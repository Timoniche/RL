from matplotlib import pyplot as plt
from tqdm import tqdm

from dqn import DQN
from params import env
from utils import generate_time_id
from video_recorder import record_video


def save_rewards(rewards):
    timeid = generate_time_id()
    with open(f'rewards_{timeid}.txt', 'w') as f:
        for r in rewards:
            f.write(f'{r} ')


def train(dqn: DQN):
    episodes = 400
    print("Need to collect (actions, states, rewards, next_statex)....")
    rewards = []
    for i in tqdm(range(episodes)):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state, env)
            next_state, reward, done, _, _ = env.step(action)

            dqn.store_transition(state, action, reward, next_state, done)
            episode_reward += reward

            if dqn.ready_to_learn():
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))
            if done:
                break
            state = next_state
        rewards.append(episode_reward)
    save_rewards(rewards)


def main():
    dqn = DQN()
    train(dqn)
    # record_video(dqn, env)


if __name__ == '__main__':
    main()
