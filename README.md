# Pong_with_PPO_Agent
Deep RL Agent using Proximal Policy Optimization for solving the Pong game.

### 1. Summary

In this project, we will train an Deep RL Agent to play the Atari game pong using Proximal Policy Optimization (PPO) algorithm. The environment is provided by OpenAI. The Agent percieves the world through pixels, and a Convolutional Neural Network is used for training the Agent. The code in this repo should be self-contained, apart from a few dependencies that are installed dynamically using pip.

The architecture of the Conv net is based in 2 Conv layers and 2 fully connected layer with sigmoid ouput:

```python
# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        
        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride 
        # (round up if not an integer)

        n_filters_1 = 4
        n_filters_2 = 16
        
        # output = 4x20x20 here
        self.conv_1 = nn.Conv2d(in_channels=2, out_channels=n_filters_1, kernel_size=6, stride=2, bias=False)
        # size=n_filters_1*40*40
        
        
        self.conv_2 = nn.Conv2d(in_channels=n_filters_1, out_channels=n_filters_2, kernel_size=6, stride=4)
        # size=n_filters_2*20*20
        
        self.size=n_filters_2*9*9
        
        # fully connected layer / 6 actions
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
   
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))

        # flatten the tensor
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        
        return self.sig(self.fc2(x))
```

#### Action Space
The Agent can choose from six different actions:
- NOOP
- FIRE
- LEFT
- RIGHT
- LEFTFIRE
- RIGHTFIRE

However, it is sufficient to train the Agent using only the actions LEFTFIRE and RIGHTFIRE. 

#### Observation Space
The size of the observation space determined by two temproally adjacent, cropped, downscaaled 80x80 greyscale screenshots of the game screen.

#### Rewards
The reward is given by the game score.

### 2. Running the code
The code is this repository requires numpy, PyTorch, OpenAI and Jupyter. Make sure that those are installed, some of the dependencies are installed directly using pip. Then, just open the Notebook 'pong-PPO.ipynb'. Follow the instructions within the notebook to train the Agent.

#### 3. Results

Without any training, the agent loses the game always.

![alt-text]()

With 1000 epochs of training, the agent wins the game!

![alt-text]()
