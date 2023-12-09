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

# plot current policy vs trend
# Updated plot function to accept two sets of episode numbers and scores
def plotTrend(episode):
    plt.figure(figsize=(10, 5))
    plt.title('Episode Score vs. Episode Number')
    plt.xlabel('Episode Number')
    plt.ylabel('Score')
    plt.plot(episode_numbers, scores, marker='o', linestyle='-', label='Score Trend')
    z1 = np.polyfit(episode_numbers, scores, 1)
    p1 = np.poly1d(z1)
    plt.plot(episode_numbers, p1(episode_numbers), "g--", label='Regression Line')
    intercept1 = z1[1]
    regression_text1 = f'Regression: y = {z1[0]:.2f}x + {intercept1:.2f}'
    corr_coefficient1, _ = pearsonr(episode_numbers, scores)
    annotation_text1 = f'{regression_text1}\nCorrelation Coefficient: {corr_coefficient1:.2f}'
    plt.annotate(annotation_text1, xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12, multialignment='left')

    plt.legend()
    plt.grid(True)
    plt.savefig(f'./trends/2000learn_e{episode}.png')

def plotAvg(episode):
    plt.figure(figsize=(10, 5))
    plt.title('Avg Episode Score vs. Episode Number')
    plt.xlabel('Episode Number')
    plt.ylabel('Avg Score')
    plt.plot(avgEpisodeNumber, avgEpisodeScore, marker='o', linestyle='-', label='Avg Score Trend')
    z1 = np.polyfit(avgEpisodeNumber, avgEpisodeScore, 1)
    p1 = np.poly1d(z1)
    plt.plot(avgEpisodeNumber, p1(avgEpisodeNumber), "g--", label='Regression Line')
    intercept1 = z1[1]
    regression_text1 = f'Regression: y = {z1[0]:.2f}x + {intercept1:.2f}'
    corr_coefficient1, _ = pearsonr(avgEpisodeNumber, avgEpisodeScore)
    annotation_text1 = f'{regression_text1}\nCorrelation Coefficient: {corr_coefficient1:.2f}'
    plt.annotate(annotation_text1, xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12, multialignment='left')

    plt.legend()
    plt.grid(True)
    plt.savefig(f'./avgTrends/2000learn_e{episode}.png')

def plotLearnedTrend(episode):
    plt.figure(figsize=(10, 5))
    plt.title('Episode Score vs. Episode Number')
    plt.xlabel('Episode Number')
    plt.ylabel('Score')
    plt.plot(episode_numbers_policy, scores_policy, marker='o', linestyle='-', label='Learned Policy Trend')
    z2 = np.polyfit(episode_numbers_policy, scores_policy, 1)
    p2 = np.poly1d(z2)
    plt.plot(episode_numbers_policy, p2(episode_numbers_policy), "r--")
    intercept2 = z2[1]
    regression_text2 = f'Regression (Learned Policy Trend): y = {z2[0]:.2f}x + {intercept2:.2f}'
    corr_coefficient2, _ = pearsonr(episode_numbers_policy, scores_policy)
    annotation_text2 = f'{regression_text2}\nCorrelation Coefficient: {corr_coefficient2:.2f}'
    plt.annotate(annotation_text2, xy=(0.7, 0.8), xycoords='axes fraction', fontsize=12, multialignment='left')

    plt.legend()
    plt.grid(True) 
    plt.savefig(f'./learned/2000learn_e{episode}.png')
agent = DQNAgent(state_size, action_size)
agent.model.load_weights('./weights.h5')
num_episodes = 1500
max_steps_per_episode = 10000
batch_size = 32
is_rendering = False
avgRewardTot = 0
for e in range(num_episodes):
    env = gym.make('SpaceInvaders-v0')
    state = env.reset()
    screen_data = state[0] 
    state_size = screen_data.shape
    state = np.reshape(screen_data, [1, state_size[0], state_size[1], state_size[2]])

    total_reward = 0

    for myTime in range(max_steps_per_episode):
        action = agent.act(state)
        results = env.step(action)
        next_state, reward, done, info = results[:4]
        next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
        if done: 
            reward -= death_penalty 
        total_reward += reward
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    avgRewardTot += total_reward
    episode_numbers.append(e+1)
    scores.append(total_reward)
    if (e+1) % 50 == 0:
        avgEpisodeNumber.append(e+1)
        avgEpisodeScore.append(avgRewardTot/50)
        if(len(avgEpisodeNumber) >= 2):
            plotAvg(e+1)
        avgRewardTot = 0

    print(f"Episode: {e+1}/{num_episodes}, Score: {total_reward}, terminated: {done}")

    if e != 0 and ((e+1) == 10 or (e+1)%50 == 0):
        for i in range(5):     
            env = gym.make('SpaceInvaders-v0', render_mode = "rgb_array")
            videoFolder =  f'./2000learn/episode_{e+1}_{i}'
            env = RecordVideo(env, f'./video/{videoFolder}', episode_trigger = lambda e: True)
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
            print(f"Current Policy for episode: {e+1}/{num_episodes}, Score: {total_reward}, terminated: {done}")

            env.close()
            if(i == 0):
                episode_numbers_policy.append(e+1)
                scores_policy.append(total_reward/5)
            else:
                scores_policy[len(scores_policy)-1] += total_reward/5
            plotTrend(e+1)
        if(len(episode_numbers_policy) >= 2):
            plotLearnedTrend(e+1)
        is_rendering = False
    # Train the agent with experiences in replay memory
    if len(agent.memory) > batch_size and is_rendering == False:
        agent.replay(batch_size)
agent.model.save_weights('./weights2000.h5')



env.close()
