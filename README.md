# Optimal Policy Learning in the Gridworld Environment: A Tabular Reinforcement Learning Approach

Our implementation focuses on addressing the problem of learning optimal policies in a grid-world environment using tabular Q-learning and SARSA algorithms. These algorithms are implemented and trained using the OpenAI [Gymnasium library](https://pypi.org/project/gymnasium/) in a grid-world environment called [Minigrid](https://github.com/Farama-Foundation/Minigrid). The grid world environment presents a classic reinforcement learning task where an agent navigates a grid, aiming to reach a goal state while avoiding obstacles. The agent’s state representation is based on partial observations, which capture the objects in the tiles surrounding the agent as demonstrated below.

![](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid-text.png)
![](https://github.com/Farama-Foundation/Minigrid/blob/master/figures/door-key-curriculum.gif)

