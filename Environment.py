import gym
from gym import wrappers, logger

# env = gym.make('CarRacing-v0')
# observation_space = env.reset()

# for _ in range(500):
#     env.render()
#     env.step(env.action_space.sample())
# def drive(neural_net):
#     reward = 0
#     observation = env.reset()
#     observation = observation.reshape([27648,1])
#     for _ in range(500):
#         env.render()
#         # print(env.action_space.shape)
#         # x = env.observation_space
#         action = neural_net.activate(observation)
#         # action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         observation = observation.reshape([27648, 1])
#         print(reward, _)
#     return observation, reward
#     pass

