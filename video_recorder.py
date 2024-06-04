import gym

from dqn import DQN


def record_video(dqn: DQN, environment):
    environment = gym.wrappers.record_video.RecordVideo(environment, "./video")
    state, _ = environment.reset()

    total_reward = 0
    while True:
        # environment.render()
        action = dqn.choose_action(state)
        next_state, reward, done, _, _ = environment.step(action)
        total_reward += 1
        state = next_state
        if done:
            environment.close()
            break
    print(f'Total reward: {total_reward}')
