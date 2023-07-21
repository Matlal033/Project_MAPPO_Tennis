import sys
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

from model import Actor, Critic
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from agent import Agent
from agent import collect_trajectories

def play_episode(agent, env, brain_name, num_agents):
    env_info = env.reset(train_mode=True)[brain_name]   #Reset at each new episode
    states = env_info.vector_observations               # get the current state (for each agent)
    scores = np.zeros(num_agents)                       # initialize the score (for each agent)

    while True:
        
        actions, _, _, _ = agent.select_action(states, False)
        env_info = env.step(actions.cpu().numpy())[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations              # get next state (for each agent)
        rewards = env_info.rewards                              # get reward (for each agent)
        dones = env_info.local_done                             # see if episode finished
        scores += env_info.rewards                              # update the score (for each agent)
        states = next_states                                    # roll over states to next time step
        if np.any(dones):                                       # exit loop if episode finished
            break

    # get the max reward between the 2 agents
    return np.max(scores)

if __name__ == "__main__":

    mean_score_tresh_to_save = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    episode = 50000
    discount_rate = .99
    tau = 0.95
    surrogate_clip = 0.2
    surrogate_clip_decay = 1
    beta = 1e-2 #entropy coefficient
    beta_decay = 1
    tmax = 400000 #timesteps while collecting trajectories
    SGD_epoch = 4
    learning_rate = 1e-4
    adam_epsilon = 3e-4
    batch_size = 128
    hidden_size = 128
    gradient_clip = 5
    rollout_size = 1280

    env = UnityEnvironment(file_name='Tennis_Windows_x86_64\Tennis.exe')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    qty_agents = 2

    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)      # initialize the score (for each agent)
    scores_window = deque(maxlen=100)  # last 100 scores

    mean_rewards = []

    agent = Agent(action_size, state_size, device, episode=episode, discount_rate=discount_rate,
            tau=tau, surrogate_clip=surrogate_clip, beta=beta, tmax=tmax, SGD_epoch=SGD_epoch, qty_agents=qty_agents,
            learning_rate=learning_rate, adam_epsilon=adam_epsilon, batch_size=batch_size, hidden_size=hidden_size,
            num_agents=num_agents, gradient_clip=gradient_clip)

    #Load previously saved network weights
    try:
        filename_actor = sys.argv[1]
        filename_critic = sys.argv[1]
    except:
        filename_actor = None
        filename_critic = None

    if filename_actor and filename_critic:
        print('Loading')
        checkpointActor = torch.load(filename_actor)
        checkpointCritic = torch.load(filename_critic)        
        agent.Actor.load_state_dict(checkpointActor)
        agent.Critic.load_state_dict(checkpointCritic)

    else:
        print('Initializing new networks')

    #prepare text file to save scores
    with open("scores.txt", "w") as file1:
        # Writing data to a file
        file1.write("Episode,Score,MeanScore100\n")
        
    n_steps = 0
    learning_iteration = 0
    for ep in range(episode):
        # collect trajectories
        env_info = env.reset(train_mode=True)[brain_name] #Reset at each new episode
        states = env_info.vector_observations             # get the current state (for each agent)
        dones = np.zeros((num_agents,1))
        while not np.any(dones):

            n_steps += 1
            actions, log_probs_old, values, concat_obs = agent.select_action(states, True)
            env_info = env.step(actions.cpu().numpy())[brain_name]  # send all actions to the environment
            next_states = env_info.vector_observations              # get next state (for each agent)
            rewards = env_info.rewards                              # get reward (for each agent)

            dones = np.array([1 if d else 0 for d in env_info.local_done])      # see if episode finished
            agent.trajectories.add_to_trajectory(states, actions, log_probs_old, rewards, values, 1-dones, concat_obs)

            if n_steps % rollout_size == 0:
                print('n_steps:', n_steps)
                agent.learn(surrogate_clip, beta)
                learning_iteration += 1
            states = next_states # roll over states to next time step

        # _, _, values, _ = agent.select_action(states)
        # agent.trajectories.add_to_trajectory(states, None, None, None, values, None, None)

        #Play episode with new policy and get score
        ep_max_score = play_episode(agent, env, brain_name, num_agents)

        # get the max reward between 2 agents
        mean_rewards.append(ep_max_score)
        scores_window.append(ep_max_score)

        #Store the score of actual and the last 100 episodeS
        # print('Episode ', ep+1,' avg score: ', ep_max_score)

        # the clipping parameter reduces as time goes on (deactivated)
        surrogate_clip*=surrogate_clip_decay

        # this reduces exploration in later runs (deactivated)
        beta*=beta_decay

        # display average over last 100 episodes every 10 episodes
        if ((ep+1) % 10)==0:
            # print('episode: ', ep+1)
            print('mean last 100 scores: ', np.mean(scores_window))
            
        #save scores to file
        with open("scores.txt", "a") as file1:
            file1.write(str(ep+1) + ","+ str(ep_max_score) + "," + str(np.mean(scores_window)) +"\n")

        if ep_max_score>mean_score_tresh_to_save:
            mean_score_tresh_to_save = ep_max_score #updating treshold score to beat before next save
            print('Episode ', ep+1,' avg score: ', ep_max_score)
            print('Saving actual weights...')
            torch.save(agent.Actor.state_dict(), "checkpoints\checkpoint_temp_actor.pth")
            torch.save(agent.Critic.state_dict(), "checkpoints\checkpoint_temp_critic.pth")
        if np.mean(scores_window)>0.5 \
        and ep+1 >= 100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format((ep+1)-100, np.mean(scores_window)))
            torch.save(agent.ActorCritic.state_dict(), 'checkpoint_temp_actor_critic.pth')
            break
