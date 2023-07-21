# Project MAPPO tennis

### Project Details

This is the third project of the Udacity Deep Reinforcement Learning nanodegree, dealing with multi-agents reinforcement learning for collaboration and competition scenarios.
For this project, the Tennis environment was solved using a Multi-Agent PPO (MAPPO) algorithm.

The essence of this problem is to have two agents, each one controlling a racket, bounce a ball over the net between themselves for as long as possible.
A reward of 0.1 is obtained when an agent bounces the ball over the net, and -0.1 if the ball hits the ground or goes out of bounds.

The main specifications for each agent are: \
State size: 8 (information about position & velocity of both the ball and racket) \
Action size: 2 (Continuous left or right movement, and jump) \

During training, the last 3 states are always stacked, in order to infer trajectories.
While training the actor, each agent only has access to its own observation space.
While training the critic, both agents have access to the observations of both agents, to better assess the state of the game as a whole.
Both agents share the same neural networks.

The environment is considered solved when the average score reaches 0.5 over 100 consecutive episodes.
In an episode, the maximum score between the two agent is the one used.

![](images/tennis_mappo_1.gif)

### Getting started

To run this code, Python 3.6 is required, along with the dependencies found in [requirements.txt](requirements.txt). \
Creating a virtual environment with those specifications is recommended.

Here is a link for documentation on how to create a virtual environment : 
[Creation of virtual environments](https://docs.python.org/3/library/venv.html)

You will also need to download the unity environment compressed file from one of the following links, and extract it under the `Project_PPO_Reacher_Continuous_Control/` folder:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

### Instructions

#### To train the agent from scratch

In *main.py* file, make sure the path to the UnityEnvironment points to *Tennis.exe*. \
E.g.: `env = UnityEnvironment(file_name='Tennis_Windows_x86_64\Tennis.exe')` \
Then, launch `main.py` from the command window.

### To train the agent from a previous checkpoint
In the command window, pass as two arguments the two file paths to the respective checkpoints of the actor and critic. \
E.g.: `main.py "checkpoints\checkpoint_temp_actor.pth" "checkpoints\checkpoint_temp_critic.pth"`

#### To watch a trained agent

First, in the *watch_agent.py* file, make sure the path to the UnityEnvironment points to *Tennis.exe*. \
Then, from the command window, launch *watch_agent.py* file. By default it will use the best checkpoints provided in the repo.

E.g.: `watch_agent.py`

To select specific weights, pass as arguments the file paths to the actor and critic checkpoints. \

E.g.:   `watch_agent.py "checkpoints\checkpoint_temp_actor.pth" "checkpoints\checkpoint_temp_critic.pth"`

### References
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/pdf/2103.01955.pdf)
- [Designing Multi-Agents systems](https://huggingface.co/learn/deep-rl-course/unit7/multi-agent-setting?fw=pt)
