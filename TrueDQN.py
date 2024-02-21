import torch
from torch import nn 
import torch.nn.functional as F
from torch.nn import Conv2d,ReLU,Sigmoid
class DQNet(nn.Module):
    def __init__(self,state_dim=(160,160),output_dim=2,batch_size=32):
        super(DQNet,self).__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(8,8),stride=4)
        self.mxpl1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(4,4),stride=2)
        self.mxpl2 = nn.MaxPool2d(kernel_size=(2,2))
        #TOBE fixed later
        self.calc_feature_map()

        self.fc1 = nn.Linear(128,256)
        self.fc2 = nn.Linear(256,output_dim)
        
    def calc_feature_map(self):
        h = (self.state_dim[0] - self.conv1.kernel_size[0] + self.conv1.padding[0] * 2) // self.conv1.stride[0] + 1 
        w = (self.state_dim[1] - self.conv1.kernel_size[1] + self.conv1.padding[1] * 2) // self.conv1.stride[1] + 1 

        h = (h - self.conv2.kernel_size[0] + self.conv2.padding[0] * 2) // self.conv2.stride[0] + 1 
        w = (w - self.conv2.kernel_size[1] + self.conv2.padding[1] * 2) // self.conv2.stride[1] + 1 

        h = (h - self.mxpl1.kernel_size[0] ) // self.mxpl1.stride[0] + 1 
        w = (w - self.mxpl1.kernel_size[1] ) // self.mxpl1.stride[1] + 1 

        h = (h - self.mxpl2.kernel_size[0] ) // self.mxpl2.stride[0] + 1 
        w = (w - self.mxpl2.kernel_size[1] ) // self.mxpl2.stride[1] + 1 

        self.feature_map = h,w
       
    def forward(self,state):
        x = F.relu(self.conv1(state))
        x = self.mxpl1(x)
        x = F.relu(self.conv2(x))
        x = self.mxpl2(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        q_values = F.softmax(self.fc2(x),dim=1)
        return q_values