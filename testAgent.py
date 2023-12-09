import gym
import numpy as np
from agent import DQNAgent
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from gym.wrappers.record_video import RecordVideo
import time
env = gym.make('SpaceInvaders-v0')
env.metadata['render.fps'] = 30
state_size = env.observation_space.shape
action_size = env.action_space.n
episode_numbers = []
scores = []
episode_numbers_policy = []
scores_policy = []
avgEpisodeNumber = []
avgEpisodeScore = []
max_time = 120  
death_penalty = 50
agent = DQNAgent(state_size, action_size)
agent.model.load_weights('./weights.h5')

env = gym.make('SpaceInvaders-v0', render_mode = "rgb_array")
env.metadata['render.fps'] = 30
videoFolder =  f'./iter3/test'
env = RecordVideo(env, f'./{videoFolder}', episode_trigger = lambda e: e % 5 == 0)
is_rendering = True
state = env.reset()
screen_data = state[0]  # This is the screen data
state_size = screen_data.shape
state = np.reshape(screen_data, [1, state_size[0], state_size[1], state_size[2]])
total_reward = 0
done = False
start_time = time.time()
while not done:
    action = agent.act(state, use_epsilon=False)
    results = env.step(action)
    next_state, reward, done, info = results[:4]
    next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])

    state = next_state
    if done: 
        reward -= death_penalty 
    total_reward += reward
    if done or time.time() - start_time > max_time:
        break  
print(f"Score: {total_reward}, terminated: {done}")

env.close()