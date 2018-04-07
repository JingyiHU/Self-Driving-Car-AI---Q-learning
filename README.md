<p align="center"><b>we use PyTorch+Kivy to run the script，environment set up tutorial https://www.superdatascience.com/pytorch/</b></p>


&nbsp;&nbsp;&nbsp;learning=>![learning](https://user-images.githubusercontent.com/22739177/32823936-c279686a-c993-11e7-906e-ea3e7830e275.gif)&nbsp;&nbsp;&nbsp;after a while=>
![after learning](https://user-images.githubusercontent.com/22739177/32823937-c2950e80-c993-11e7-9358-89e50cdaae8f.gif)


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

Then we combine the Q-learning algorithm with deep learning. From the High Level perspective, Q-learning has achieved the basic function of unmanned vehicles to avoid roadblocks, and the deep learning algorithm allows the vehicle to automatically summarize and learn features, reducing the incomplete nature of human-made features to better adapt Very complicated environmental conditions.


![learn](https://user-images.githubusercontent.com/22739177/32822235-60bfc1b6-c98c-11e7-966a-2a2c295645cc.PNG)
The calculated L value is back-propagated to calculate the weight w of each synapse (green circle) so that the L value can be as small as possible.

It should be noted that the above process is called "learning". 

![act](https://user-images.githubusercontent.com/22739177/32822234-60a7c57a-c98c-11e7-82b2-82d53104940a.PNG)
The process of determining the "action" is the process of passing the obtained Q value to "Softmax-Function". "Softmax-Function" is an action selection strategy. It can help us make the best choice based on the current data. 
