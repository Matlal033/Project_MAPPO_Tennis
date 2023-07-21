import sys
import torch
import numpy as np
from model import Actor, Critic
from unityagents import UnityEnvironment
from agent import Agent
from agent import collect_trajectories

if __name__ == "__main__":

    mean_score_tresh_to_save = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    episode = 200000
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
    batch_size = 128 #64 #500
    hidden_size = 128
    gradient_clip = 5
    rollout_size = 1280 #640 #500

    env = UnityEnvironment(file_name='Tennis_Windows_x86_64\Tennis.exe')

   # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations #20 in parallel
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    qty_agents = 2

    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)      # initialize the score (for each agent)


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
        print('Loading default checkpoints')
        checkpointActor = torch.load("checkpoint_temp_actor_best.pth")
        checkpointCritic = torch.load("checkpoints\checkpoint_temp_critic_best.pth")
        agent.Actor.load_state_dict(checkpointActor)
        agent.Critic.load_state_dict(checkpointCritic)


    env_info = env.reset(train_mode=False)[brain_name] #Reset at each new episode
    states = env_info.vector_observations                  # get the current state (for each agent)                       # initialize the score (for each agent)
    dones = np.zeros((num_agents,1))
    while not np.any(dones):

        actions, log_probs_old, values, concat_obs = agent.select_action(states, True)
        env_info = env.step(actions.cpu().numpy())[brain_name]  # send all actions to the environment
        next_states = env_info.vector_observations              # get next state (for each agent)
        rewards = env_info.rewards                              # get reward (for each agent)

        dones = np.array([1 if d else 0 for d in env_info.local_done])#env_info.local_done                        # see if episode finished

        states = next_states # roll over states to next time step
