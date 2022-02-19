import gym
import ale_py
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Multiply
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# print('gym: ', gym.__version__)
# print('ale_py', ale_py.__version__)
#Michael's Copy

# env = gym.make('Breakout-ram-v0')
env = gym.make("Breakout-ram-v0")

observations = env.observation_space.shape[0]
actions = env.action_space.n

def build_model(observations, actions):
  model = Sequential()
  model.add(Flatten(input_shape=(1, observations)))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(actions, activation='linear'))
  return model


model = build_model(observations, actions)

def build_agent(model, actions):
  policy = BoltzmannQPolicy()
  memory = SequentialMemory(limit=1000, window_length=1)
  dqn = DQNAgent(model=model, memory=memory, nb_actions=actions)
  return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=5e-2))
dqn.fit(env, nb_steps=100000, visualize=True)



# for episode in range(40):
#   observation = env.reset()
#   for t in range(100):
#     env.render()
#     #print(observation)
#     action = env.action_space.sample()
#     print(action)
#     observation, reward, done, info = env.step(action)
#     if t%50 == 0 :
#       print("Observation: ", observation)
#     if done:
#       print("Episode finished after {} timestamps".format(t+1))
#       break
#     time.sleep(.1)
# env.close()
