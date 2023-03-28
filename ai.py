import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.autograd as Variable


#creating architecture
#implementing init function
class Network(nn.Module):    #nn.Module is used by inheritance
    def __init__(self, input_size, nb_action):  #5 input neurons because of 3 sensors + orientation and -orientation  ; nb_action will be three
        super(Network,self).__init__()
        self.input_size=input_size
        self.nb_action=nb_action
        self.fc1=nn.Linear(input_size, 30)  #fc1 means full connection between input layer and hidden layer it has 2 parameters input neurons and output neurons , here 30 is chosen through experience
        self.fc2=nn.Linear(30, nb_action)  #fc2 is the full connection between hidden layer and output layer  it will have 30 as input and output will be nbaction
        
        
    def forward(self,state): #forward function will have 2 inputs as self which is to use variables of object and state is used to give the state ; the output will have q-values of three action-left,right,forward
        x=F.relu(self.fc1(state))    #relu is rectifier function  ; x represents hidden nurons and we use relu to activate them , here we are using fc1
        q_values=self.fc2(x)
        return q_values
    
    
    
    
#implementing Experience Replay
class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity=capacity #capacity is max no. of transitions we want in memory
        self.memory=[]
        
    def push(self,event):  #event comprises of last state, new state, last action and reward
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]
        
    def sample(self,batch_size): #batch size  is the sample size
        samples=zip(*random.sample(self.memory,batch_size)) #zip reshapes the list, here we are selecting random samples from memory
        return map(lambda x: Variable(torch.cat(x,0)), samples)   #too difficult just know that it is used to sample the memory, it is used to make use of pytorch Variable
        
    
    
#implementing Deep-Q learning
class Dqn():
    def __init__(self,input_size,nb_action,gamma): #gamma is the delay coefficent
        self.gamma=gamma
        self.reward_window=[]
        self.model=Network(input_size, nb_action)
        self.memory=ReplayMemory(100000)
        self.optimizer=optim.Adam(self.model.parameters(),lr=0.001)# lr is learning rate,value should be not high
        self.last_state=torch.Tensor(input_size).unsqueeze(0)
        