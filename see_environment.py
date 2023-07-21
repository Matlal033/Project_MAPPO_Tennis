from unityagents import UnityEnvironment
import numpy as np


def see_environment():
    env = UnityEnvironment(file_name='Tennis_Windows_x86_64\Tennis.exe')

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[0]
    # print('There are {} agents. Each observes a state with length: {}'.format(states.shape[1], state_size))
    # print('The state for the first agent looks like:', states[0])
    # print('The state for the second agent looks like:', states[1])
    # print('The state e:', states)
    concat_obs = [*states[0],*states[1]] # Concatenate the obs of both agents
    # print('The concatenated states for both agents look like:', concat_obs)

    for i in range(15):                                         # play game for 15 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        step = 0
        while True:
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            
            if np.any(dones):                                  # exit loop if episode finished
                print('step: ', step)
                break
            step += 1
        print('Total score per agent this episode: {}'.format((scores)))
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
        print('Max score (between 2 agents) this episode: {}'.format(np.max(scores)))

if __name__ == "__main__":
    see_environment()
