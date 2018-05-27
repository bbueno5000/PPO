import gym
import logging

if __name__ == '__main__':
    env = gym.make('VerticalLanding-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        logging.info('Action:{}'.format(action))
        logging.info('Observations Size:{}'.format(obs.shape))
        logging.info('Score:{}'.format(reward))
        if done:
            break
