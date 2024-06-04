from statistics import mean
from tqdm import tqdm

from dqn import DQN
from params import env
from utils import generate_time_id
from video_recorder import record_video


def save_metrics(
        metrics,
        nameprefix,
):
    timeid = generate_time_id()
    with open(f'metrics/{nameprefix}_{timeid}.txt', 'w') as f:
        for r in metrics:
            f.write(f'{r} ')


def train(dqn: DQN):
    episodes = 400
    print("Need to collect (actions, states, rewards, next_statex)....")
    rewards = []
    episode_losses = []
    for i in tqdm(range(episodes)):
        state, _ = env.reset()
        episode_reward = 0
        losses = []
        while True:
            env.render()
            action = dqn.choose_action(state, env)
            next_state, reward, done, _, _ = env.step(action)

            dqn.store_transition(state, action, reward, next_state, done)
            episode_reward += reward

            if dqn.ready_to_learn():
                loss = dqn.learn()
                losses.append(loss)
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))
            if done:
                break
            state = next_state
        rewards.append(episode_reward)
        if len(losses) == 0:
            episode_losses.append(0)
        else:
            losses = list(map(lambda tensor: tensor.item(), losses))
            episode_losses.append(mean(losses))

    save_metrics(rewards, nameprefix='rewards')
    save_metrics(episode_losses, nameprefix='losses')


def main():
    dqn = DQN()
    train(dqn)
    # record_video(dqn, env)


if __name__ == '__main__':
    main()
