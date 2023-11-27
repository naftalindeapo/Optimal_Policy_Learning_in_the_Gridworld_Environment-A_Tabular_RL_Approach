# Optimal Policy Learning in the Gridworld Environment: A Tabular Reinforcement Learning Approach

Our implementation focuses on addressing the problem of learning optimal policies in a grid-world environment using tabular Q-learning and SARSA algorithms. These algorithms are implemented and trained using the OpenAI [Gymnasium library](https://pypi.org/project/gymnasium/) in a grid-world environment called [Minigrid](https://github.com/Farama-Foundation/Minigrid). The grid world environment presents a classic reinforcement learning task where an agent navigates a grid, aiming to reach a goal state while avoiding obstacles. The agent’s state representation is based on partial observations, which capture the objects in the tiles surrounding the agent as demonstrated below.

<div>
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid-text.png" style="width: 40%; float: left;" /> 
    <img src="https://github.com/Farama-Foundation/Minigrid/blob/master/figures/door-key-curriculum.gif" style="width: 50%; float: right;" />  
 </div>

# The Environment
The grid world environment used in our implementation was written by [Maxime Chevalier-Boisvert](https://github.com/Farama-Foundation/Minigrid). This environment is an empty room, and the agent's goal is to reach the green goal square, which provides a sparse reward.

### Structure of the world
In general, the structure of the grid world environment is as follows:
- The world is an NxM grid of tiles
- Each tile in the grid world contains zero or one object
- Cells that do not contain an object have the value None
- Each object has an associated discrete colour (string)
- Each object has an associated type: 'unseen': 0,
'empty': 1, 'wall': 2, 'floor': 3, 'door': 4, 'key': 5, 'ball': 6, 'box': 7, 'goal': 8.

The agent can pick up and carry exactly one object (e.g., a ball or key). To open a locked door, the agent has to carry a key matching the door's colour. In our implementation, we used the empty 8-by-8 Minigrid environment ('MiniGrid-Empty-8x8-v0').

### Actions in the basic environment
In our implementation, the agent can only perform three basic actions: ‘Turn left’, ‘Turn right’, and ‘Move forward’. To explore the environment, we will train the agent to select an action to take randomly. During training, we reset the environment after each episode has ended to set the number of steps the agent has taken to zero and put the agent back in its starting position. We will also use this resetting to obtain the first observation representing the current state $S$.

### Observation returned by Environment
We use a wrapper provided by the minigrid environment to only extract the observation representing the partial view that the agents have of the environment. The ’image’ observation contains information about each tile around the agent. Each tile is encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE), however, for this task, we only used the information contained in the OBJECT_IDX.

### Reward Signal
The empty environment we are using provides a sparse reward. The agent only receives a non-zero reward when the goal is successfully achieved. The reward upon success is determined by
$$r = 1 − 0.9 ×\frac{\text{number of steps}}{\text{maximum number of steps}}$$

The maximum number of steps in the empty environment is $4 \times N \times M$ where $N\times M$ is the size of the grid. For an $8\times 8$ empty grid world environment we are using, the maximum number of steps is $256$ steps.

### Variables returned from step() function
During the training, the environment returns five variables from the step() function:
$\verb|[observation, reward, done, truncated|$, $\verb|info] = env.step(...)|,$
where,
- $\verb|observation|$ - the observation returned by the environment.
- $\verb|reward|$ - the scalar reward associated with taking the action and reaching the next state.
- $\verb|done|$ - a boolean variable indicating if the agent reached a terminal state.
- $\verb|truncated|$ - a boolean variable indicating if the agent did not manage to reach a terminal state, but the episode ended because the maximum number of steps was reached.
- $\verb|info|$ - contains auxiliary diagnostic information.

# Training the agent
In our implementation, we trained the agent in the grind world environment using tabular Q-learning and State-Action-Reward-State-Action (SARSA) algorithms. The formal algorithms can be found in [Sutton and Barto](http://incompleteideas.net/book/the-book.html) on pages 131 and 136. The full details about the training process, including hyperparameter tuning, can be found in the formal paper I have written, which can be found [here](https://drive.google.com/uc?id=1OtTvD8HqZqN_MSK_1CczTXJZiSk-1o9G).

### Optimal parameters
The Table below shows the optimal parameters used to train the agent in the grid world environment, obtained using the grid search.
| Parameters | $\alpha$ |$\gamma$|$\epsilon_\text{min}$|$\epsilon_\text{max}$ | $\epsilon_\text{decay}$|
| :------------|:------------|:------------|:----------|:------------|:------------|
| Values       | $0.1$       | $0.9$       |$0.01$     |$1.0$        |$0.995$      |

# Results
At the end of each episode, we add both the reward and the total number of training steps to the tensorboard writer. Both algorithms were trained for 3000 episodes in the 8-by-8 grid world environment with 256 steps. Figure 1 below shows the total number of steps accumulated at each episode, and the total rewards were monitored in the tensorboard during training.
![alt text](https://drive.google.com/uc?id=1nhCTEJfRpTh99JNDGfumlF9QzTZyN2na) 

### Evaluation results
After training, the saved Q-tables were used to evaluate the performance of the agents for 1000 evaluation episodes. The below shows the evaluation results.

|               | Q-learning  |SARSA        |
| :-------------|:------------|:------------|
|Average steps  |$12.0$       |$11.0$       |
|Completion rate (\%)|$100$   |$100$        |
|Average reward |$0.9578$     |$0.9613$     |


# Installation
To install the Minigrid library use pip install minigrid.  

### Dependencies
- Anaconda 3
- Python 3.7-3.12

### License
Refer to the [LICENSE](https://github.com/naftalindeapo/Optimal_Policy_Learning_in_the_Gridworld_Environment-A_Tabular_RL_Approach/blob/main/LICENSE).


