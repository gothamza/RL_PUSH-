import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from TrueDQN import DQNet
from collections import deque
import numpy as np
import random
from preprocessing import RGBToGray,Resize,Normalize
class DQNAgent:
    def __init__(self,state_dim,output_dim,lr=0.05,gama=0.99,memory_size=1000,batch_size = 32,epsilon=1,chkpt_file="DQN.pth"):
        self.lr = lr
        self.gama = gama
        self.batch_size = batch_size
        self.q_net = DQNet(state_dim,output_dim,batch_size)
        self.optimizer = Adam(self.q_net.parameters(),lr)
        self.loss_function = MSELoss()
        self.replay = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.chkpt_file = "SAVED/"+chkpt_file
    def remember(self,state,action,reward,state_,done):
        
        state = Normalize(RGBToGray(Resize(state))).reshape((84,84,1))
        state_ = Normalize(RGBToGray(Resize(state_))).reshape((84,84,1))
        
        state = np.transpose(state,(2,0,1))
        state_ = np.transpose(state_,(2,0,1))
        
        current_exp = (state,action,reward,state_,done)
        self.replay.append(current_exp)
    def act(self,state):
        if np.random.rand() < self.epsilon:
            self.action = np.random.randint(0,self.q_net.output_dim)
        else:
            state = Normalize(RGBToGray(Resize(state))).reshape((84,84,1))
            state = np.transpose(state,(2,0,1))
            self.action = np.argmax(self.q_net(torch.from_numpy(state).unsqueeze(dim=0).float()).detach().numpy())
        #print("action : ",self.action)
        return self.action

    def be_reasonable(self,epoch):
        if self.epsilon > 0.1 :
            self.epsilon = 1/epoch
        else :
            self.epsilon = 0.1
    
    def learn(self):
        if len(self.replay) > self.batch_size :
            batch = random.sample(self.replay,self.batch_size)
            states  = torch.cat([torch.from_numpy(s).float().unsqueeze(dim=0) for (s,_,_,_,_) in batch])
            actions = torch.tensor([a for (_,a,_,_,_) in batch])
            rewards = torch.tensor([r for (_,_,r,_,_) in batch])
            states_ = torch.cat([torch.from_numpy(S).float().unsqueeze(dim=0) for (_,_,_,S,_) in batch])
            dones   = torch.tensor([d for (_,_,_,_,d) in batch])
            
            Qs = self.q_net(states)
            with torch.no_grad():
                Qs_ = self.q_net(states_)
            
            Qs_targets = rewards + self.gama * (1-dones.to(torch.int)) * torch.max(Qs_,axis=1)[0]
            qs = Qs.gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()
            
            l = self.loss_function(qs,Qs_targets.detach())
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            return l.item()
        else:
            return 0.0
        
    def save(self):
        print("... saving model ...")
        torch.save(self.q_net,self.chkpt_file)

    def load(self):
        print("... loading model ...")
        self.q_net = torch.load(self.chkpt_file)
    def save_file(self,file_name):
        torch.save(self.q_net,file_name)

    def load_file(self,file_name):
        self.q_net = torch.load(file_name)