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
        self.last_action=0
        self.last_reward=0
        

    
    
    # def select_action(self, state):
    #    probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100;  #probs stand for finding probablities and softmax function is used to select action based on probablity, state is a torch tensor and we are converting it into torch variable, by using volatile=true we are specifying we dont want gradient along with graphs
    #    # 7 is the temperature parameter , it is used to maximize the softmax function
    #    action = probs.multinomial(num_samples=1)#multinomial is used to randomly select the probablitiy values
    #    return action.data[0,0]# it is a trick

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self,batch_state,batch_next_state,batch_reward,batch_action): #batch_state is for current state , used for transitions
        outputs=self.model(batch_state).gather(1,batch_action.unsqueeze[1]).squeeze(1) #we receive theoutput for all three action(0,1,2) but we want action that is chosen therefore we use gather func (1,batch_action ) chooses the best action always
        #unsqueeze is used for fake dimaension ,earlier we used unsqueeze for batch_state therefore we also have to do it for batch_action to match the dimensions
        next_outputs=self.model(batch_next_state).detach().max(1)[0]  #batch next state contains multiple next states detach functin is used to detach them and max function is applied to find max q-values , since we are finding max q-values wrt to action me do [1]
        target=self.gamma * next_outputs + batch_reward
        td_loss=F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad() #just write
        td_loss.backward(retain_variables=True)
        self.optimizer.step()
        
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")


