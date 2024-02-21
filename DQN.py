
import torch
from torch import nn
from torch.nn import Linear,ReLU,MSELoss
from torch.optim import Adam
import numpy as np 
from collections import deque
import  gymnasium as gym
import matplotlib.pyplot as plt
import cv2
import random

if __name__ == "__main__":

        torch.autograd.set_detect_anomaly(True)

        def RGBtoGray(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
        env.metadata["render_fps"] = 240
        ckpt_file = "DQN.pth"

        epochs = 1000
        losses = list()

        state_dim = (210,160)
        h1 = 120
        h2 = 120
        action_dim = 6
        memory_size = 1000
        replay = deque(maxlen=memory_size)
        batch_size = 32
        model = nn.Sequential(
            Linear(state_dim[0]*state_dim[1],h1),
            ReLU(),
            Linear(h1,h2),
            ReLU(),
            Linear(h2,action_dim)
        )

        model = torch.load('DQN.pth')

        loss_function = MSELoss()
        learning_rate = 1e-3
        optimizer = Adam(model.parameters(),lr= learning_rate)

        gama = 0.9
        epsilon = 1

        for epoch in range(1,epochs+1):
            done = False
            total_reward = 0.0
            state,_ = env.reset()
            state = RGBtoGray(state)
            state = torch.from_numpy(state).float().view(1,-1)
            loss_t = 0.0
            reward_t = 0.0
            while not done :
                
                Q = model(state)
                r = np.random.rand()
                if r < epsilon :
                    action = np.random.randint(0,6)
                else:
                    with torch.no_grad():
                        action = np.argmax(Q.numpy())
                        #print(f"action taken by the AI {action}")
                
                state_,reward,done,_,_ = env.step(action)
                reward_t += reward
                env.render()
                state_ = RGBtoGray(state_)
                state_ = torch.from_numpy(state_).float().view(1,-1)
                current_exp = (state,action,reward,state_,done)
                replay.append(current_exp)
                
                if len(replay) > batch_size :
                    batch = random.sample(replay,batch_size)
                    #To insure that the agent train on the current experience 
                    """
                    if current_exp not in batch :
                        batch.append(current_exp)
                    """
                    #print(batch)
                    states  = torch.cat([s for (s,_,_,_,_) in batch])
                    actions = torch.tensor([a for (_,a,_,_,_) in batch])
                    rewards = torch.tensor([r for (_,_,r,_,_) in batch])
                    states_ = torch.cat([S for (_,_,_,S,_) in batch])
                    dones   = torch.tensor([d for (_,_,_,_,d) in batch])
                    Qs = model(states)
                    with torch.no_grad():
                        Qs_ = model(states_)
                    
                    #MaxQs_ = torch.max(Qs_,axis=1)[0]
                    
                    Qs_targets = rewards + gama * (1-dones.to(torch.int)) * torch.max(Qs_,axis=1)[0]
                    qs = Qs.gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()
                    
                    l = loss_function(qs,Qs_targets.detach())
                    loss_t += l.item()
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                
            if epsilon > 0.1 :
                epsilon -= 1/epoch
                
            
            if (epoch % 100) ==0 :
                print(f"--------- saving the model ---------")
                torch.save(model,ckpt_file)
            
            print(f"******** loss at epoch {epoch} = {loss_t} ********")
            print(f"******** reward at epoch {epoch} = {reward_t} ********")

        plt.plot(range(epochs),losses)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Performance')
        plt.show()



















