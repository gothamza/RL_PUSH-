from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Memory import Memory,Trajectory
import torch
import numpy as np

class PPOAgent:
    def __init__(self,n_actions,input_dims,gamma=0.99,lr=0.0003,lmbda=0.95,epsilon=0.2,batch_size=64,N=2048,learn_epochs=10,condition=False):
        self.gamma =gamma 
        self.learn_epochs = learn_epochs
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.actor = ActorNetwork(n_actions,input_dims,lr,condition)
        self.critic = CriticNetwork(input_dims,lr,condition)
        self.memory = Memory(batch_size)
        self.current_trajectory = Trajectory()
    def remember_exp(self,state,action,probs,vals,reward,done):
        self.current_trajectory.add_experience(state,action,probs,vals,reward,done)
    
    def remember_tra(self):
        self.memory.add_trajectory(self.current_trajectory)
        self.current_trajectory = Trajectory()

    def save_model(self):
        print("... saving models ...")
        self.actor.save_chkpt()
        self.critic.save_chkpt()

    def load_model(self):
        print("... loading model ...")
        self.actor.load_chkpt()
        self.critic.load_chkpt()

    def act(self,observation):
        state = torch.tensor([observation],dtype=torch.float)#.to(self.actor.device)
        policy = self.actor(state)
        value = self.critic(state)
        action = policy.sample()
        log_probs = torch.squeeze(policy.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action,log_probs,value

    def learn(self):
        if self.memory.counter >= self.batch_size:
            print("begin learning")
            total_loss = 0
            for _ in range(self.learn_epochs):
                trajectories = self.memory.generate_batch(self.batch_size)
                for tra in trajectories:
                    
                    advantages = torch.zeros(tra.counter)
                    delta = 0

                    for j in reversed(range(tra.counter - 1)):
                        delta = tra.rewards[j + 1] + self.gamma * tra.values[j + 1] * (1 - int(tra.dones[j + 1])) - tra.values[j]
                        advantages[j] = delta + self.gamma * self.lmbda * advantages[j + 1]
                    for index in range(tra.counter-1):
                        old_prob = tra.probs[index]
                        new_policy = self.actor(torch.tensor(tra.states[index]))
                        new_prob = new_policy.sample()
                        ratio = new_prob / old_prob
                        #delta = tra.values[index + 1] + tra.rewards[index] - tra.values[index]
                        value = self.critic(torch.tensor(tra.states[index]))
                        value_ = self.critic(torch.tensor(tra.states[index+1]))
                        critic_loss = self.critic.loss_function(self.gamma*value_ + tra.rewards[index],value)
                        ratio = new_prob / old_prob
                        clipped_loss = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                        unclipped_loss = ratio * advantages
                        actor_loss = -torch.mean(torch.min(clipped_loss, unclipped_loss))
                        actor_loss.requires_grad=True
                        self.actor.optimizer.zero_grad()
                        self.critic.optimizer.zero_grad()
                        critic_loss.backward()
                        actor_loss.backward()
                        total_loss += critic_loss.item()
                        total_loss += actor_loss.item()
                        self.actor.optimizer.step()
                        self.critic.optimizer.step()
            return total_loss
        else:
            return 0
        




















