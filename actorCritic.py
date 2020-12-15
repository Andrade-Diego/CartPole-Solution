'''
BASED ON A TUTORIAL BY MACHINE LEARNING WITH PHIL
https://www.youtube.com/watch?v=53y49DBxz8U
'''

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, alpha, inputDims, numActions, layer1Dims = 128, layer2Dims = None, layer3Dims = None):
        #### inherits nn.Module
        super (ActorCritic, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None

        #### IF ONE LAYER
        if (layer2Dims is None) and (layer3Dims is None):
            self.layer1 = nn.Linear(inputDims, layer1Dims)

            #### THIS IS THE ACTOR
            self.pi = nn.Linear(layer1Dims, numActions)
            #### THIS IS THE CRITIC
            self.v = nn.Linear(layer1Dims, 1)


        #### IF TWO LAYER
        elif (layer2Dims is not None) and (layer3Dims is None):
            self.layer1 = nn.Linear(inputDims, layer1Dims)
            self.layer2 = nn.Linear(layer1Dims, layer2Dims)
            self.pi = nn.Linear(layer2Dims, numActions)
            self.v = nn.Linear(layer2Dims, 1)


        #### IF THREE LAYER
        elif (layer2Dims is not None) and (layer3Dims is not None):
            self.layer1 = nn.Linear(inputDims, layer1Dims)
            self.layer2 = nn.Linear(layer1Dims, layer2Dims)
            self.layer3 = nn.Linear(layer2Dims, layer3Dims)
            self.pi = nn.Linear(layer3Dims, numActions)
            self.v = nn.Linear(layer3Dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        #### FORWARD PROP FOR ONE LAYER
        if (self.layer2 is None) and (self.layer3 is None):
            x = F.relu(self.layer1(state))

        #### TWO LAYERS
        elif(self.layer2 is not None) and (self.layer3 is None):
            x = F.relu(self.layer1(state))
            x = F.relu(self.layer2(x))

        #### THREE LAYERS
        elif (self.layer2 is not None) and (self.layer3 is not None):
            x = F.relu(self.layer1(state))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))

        #### send through the policy layer and value layer to get actor and critic vals
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)

class Agent():
    def __init__(self, alpha, gamma, inputDims, numActions, layer1Dims = 128, layer2Dims = None, layer3Dims = None):
        self.gamma = gamma
        self.layer1Dims = layer1Dims
        self.layer2Dims = layer2Dims
        self.layer3Dims = layer3Dims

        self.actorCritic = ActorCritic(alpha, inputDims, numActions, layer1Dims, layer2Dims, layer3Dims)
        self.logProb = None

    def chooseAction(self, observation):

        #### converts observation into a tensor to enable forward propogation in network
        state = T.tensor([observation], dtype=T.float).to(self.actorCritic.device)

        #### this gets us the policy from actorcritic, don't care about the values from the critic right now
        policy, _ = self.actorCritic.forward(state)

        #### makes us certain we pick an action
        policy = F.softmax(policy, dim=1)

        #### turns policy into a probability distribution
        actionDistr = T.distributions.Categorical(policy)
        action = actionDistr.sample()

        #### this gets used in the cost function
        self.logProb = actionDistr.log_prob(action)

        return action.item()

    def learn(self, state, reward, newState, done):
        #### new gradients for each learning iteration
        self.actorCritic.optimizer.zero_grad()

        #### before state and newstate get forwardproped they should get turned into tensors
        state = T.tensor([state], dtype=T.float).to(self.actorCritic.device)
        newState = T.tensor([newState], dtype=T.float).to(self.actorCritic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actorCritic.device)

        #### here what we care about are the critics
        _, criticVals = self.actorCritic.forward(state)
        _, newCriticVals = self.actorCritic.forward(newState)

        delta = reward + self.gamma*newCriticVals*(1-int(done)) - criticVals

        actorLoss = -self.logProb*delta
        criticLoss = delta**2

        (actorLoss + criticLoss).backward()
        self.actorCritic.optimizer.step()
