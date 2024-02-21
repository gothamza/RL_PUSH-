import torch
from torch import nn
from torch.nn import Linear,Conv2d,AvgPool2d,ReLU
from torch.optim import Adam
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import os
from preprocessing import *
class ActorNetwork(nn.Module):
    def __init__(self,n_actions,input_dims,lr,condition,fc1_dims=256,fc2_dims=256,chkpt_dir="SAVED"):
        super(ActorNetwork,self).__init__()
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir,"actor_torch_ppo")
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.lr = lr
        self.condition = condition

        self.fc1 = Linear(input_dims,fc1_dims)#128
        self.Fc1 = Linear(128,fc1_dims)
        self.fc2 = Linear(fc1_dims,fc2_dims)
        self.output = Linear(fc2_dims,n_actions)
        self.optimizer = Adam(self.parameters(),lr=lr)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(8,8),stride=4)
        self.mxpl1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(4,4),stride=2)
        self.mxpl2 = nn.MaxPool2d(kernel_size=(2,2))
        #self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        #self.to(self.device) 

        self.optimizer = Adam(self.parameters(),lr=lr)

    def forward(self,state):
        if not self.condition:    
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            policy = F.softmax(self.output(x),dim=-1)
            policy = Categorical(policy)
            return policy
        else:
            #state = torch.tensor(state)#.unsqueeze(dim=0)
            state = Normalize(RGBToGray(Resize(state))).reshape((84,84,1))
            state = np.transpose(state,(2,0,1))
            x = F.relu(self.conv1(state))
            x = self.mxpl1(x)
            x = F.relu(self.conv2(x))
            x = self.mxpl2(x)
            x = x.view(x.size(0),-1)

            x = F.relu(self.Fc1(state))
            x = F.relu(self.fc2(x))
            policy = F.softmax(self.output(x),dim=-1)
            policy = Categorical(policy)
            return policy

    def save_chkpt(self):
        torch.save(self.state_dict(),self.chkpt_file)
    def load_chkpt(self):
        torch.load_state_dict(torch.load(self.chkpt_file))