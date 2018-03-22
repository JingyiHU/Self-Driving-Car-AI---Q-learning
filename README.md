<p align="center"><b>we use PyTorch+Kivy to run the script，environment set up tutorial https://www.superdatascience.com/pytorch/</b></p>


&nbsp;&nbsp;&nbsp;learning=>![learning](https://user-images.githubusercontent.com/22739177/32823936-c279686a-c993-11e7-906e-ea3e7830e275.gif)&nbsp;&nbsp;&nbsp;after a while=>
![after learning](https://user-images.githubusercontent.com/22739177/32823937-c2950e80-c993-11e7-9358-89e50cdaae8f.gif)


The Deep Q-learning algorithm used in this unmanned vehicle AI project is a deep reinforcement learning algorithm invented by DeepMind in 2013. The combination of the Q-learning idea and the neural network algorithm can also be considered as the source of modern reinforcement learning algorithms. The researchers used this algorithm to let the computer learn 49 Atari games in 2015 and beat humans in most of the games. From the aspect of applicability, we do not need to tell AI specific rules. As long as it is constantly explored, it can slowly find the law and accomplish many intellectual activities that were previously considered only humans could accomplish.

Since it is a combination of Q-learning and Deep learning, we can discuss what Q-learning is.

Q-learning is a reinforcement learning algorithm. Unmanned vehicles need to take actions according to the current state, and after obtaining corresponding rewards, they must improve these actions so that the next time they reach the same state, the unmanned vehicle can do excellent choice at the end. We use Q(S,A) to represent **the utility value** obtained by taking action A in the S state. In the following, the letter R is used to represent Rewards, and S' represents the new position that the car reaches after the A action is taken. (The difference between reward value R and the utility value Q is that R represents **the reward of this position**. For example, for the unmanned vehicle, the position reward for the obstacle is -100, the position reward for the river is -120, the reward for the road is 100, and the reward for the target is 10000. And Q stands for **the utility value of this action**, which is used to evaluate the merits of taking this action in a specific state. Can be understood as the brain of the unmanned vehicle. It is a comprehensive consideration of all known states.)
The pseudo code is as follows:
```
Initialize Q arbitrarily // Initialize the Q value randomly
Repeat (for each episode): // Every attempt, from the car to the crash wall is an episode
Initialize S // Starting of the vehicle, S is the state of the initial position
Repeat (for each step of episode):
Q(S,A) ← (1-α)*Q(S,A) + α*[R + γ*maxQ(S',a)] // Q-learning core Behrman equation, update action utility value
S ← S' // Update location
Until S is terminal // the location reaches the end
```

In the Bellman Equation, γ is the discount factor and α is the learning rate. The greater the gamma, the more attention will be paid to the previous experience by the unmanned vehicle, and the smaller the γ, the more attention will be paid to the immediate interest. The range of α is 0~1. The bigger the value is, the less effective the training is. It can be seen that when the value of α is 0, no matter how the AI is trained, the new Q value cannot be learned; when the value of α is 1, the new Q value will completely replace the old Q value, and each training will get a new value and it will be completely forgotten the previous training results. These parameter values are artificially set and need to be adjusted slowly based on experience.

Then we combine the Q-learning algorithm with deep learning. From the High Level perspective, Q-learning has achieved the basic function of unmanned vehicles to avoid roadblocks, and the deep learning algorithm allows the vehicle to automatically summarize and learn features, reducing the incomplete nature of human-made features to better adapt Very complicated environmental conditions.

First, use a deep neural network as a network of Q values. Each point on the map has coordinates (X1, X2). Enter this state into the neural network to predict the Q value for each direction (assuming four actions in the figure. Four directions, so get a total of four new Q values.) Q-target represents the Q value obtained when the state was last reached, and then uses the mean-square error to define the Loss Function.

![learn](https://user-images.githubusercontent.com/22739177/32822235-60bfc1b6-c98c-11e7-966a-2a2c295645cc.PNG)
The calculated L value is back-propagated to calculate the weight w of each synapse (green circle) so that the L value can be as small as possible.

It should be noted that the above process is called "learning". Although we compared the previous Q value and feed it back to the input, the Q value calculated this time is constant. What we need to do next is to make an "action" based on the Q value calculated this time.

![act](https://user-images.githubusercontent.com/22739177/32822234-60a7c57a-c98c-11e7-82b2-82d53104940a.PNG)
The process of determining the "action" is the process of passing the obtained Q value to "Softmax-Function". "Softmax-Function" is an action selection strategy. It can help us make the best choice based on the current data. The principle involves probability theory. Here we focus on the application layer. There are detailed comments in the code. See [Wiki] (https://en.wikipedia.org/wiki/Softmax_function).

So why not directly select the action corresponding to the largest Q value, but use Softmax-Function to make the decision? There are several action selection strategies involved here. It is not impossible to directly select the largest Q value. This is called greedy strategy. The disadvantage is that it is easy to fall into the local optimal solution. Because if you finally achieve the goal after performing an action, then this strategy will always choose this action in the subsequent state, resulting in no chance to explore the global optimal solution.
