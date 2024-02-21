import numpy as np
"""
class Memory:
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.states = list()
        self.actions = list()
        self.probs = list()
        self.values = list()
        self.rewards = list()
        self.dones = list()
    def generate_batch(self):
        batch_starts = np.range(0,len(self.states),self.batch_size)
        indecies = np.arange(len(self.states))
        np.random.shuffle(indecies)
        batches = np.array([indecies[i:i+self.batch_size] for i in batch_starts])
        return np.array(self.states),np.array(self.actions),np.array(self.probs),np.array(self.values),np.array(self.rewards),np.array(self.dones),batches
    def store_memory(self,state,action,prob,value,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    def clear_memory(self):
        self.states  = list()
        self.probs   = list()
        self.values  = list()
        self.rewards = list()
        self.states_ = list()
        self.dones   = list()

"""

class Memory:
    def __init__(self,batch_size=10):
        self.trajectories = list()
        self.counter = 0
    def add_trajectory(self,trajectory):
        self.trajectories.append(trajectory)
        self.counter += 1

    def delete_trajectory(self,index):
        self.trajectories.pop(index)
        self.counter -= 1

    def clear_memory(self):
        self.trajectories = list()
        self.counter = 0
    
    def generate_batch(self,size):
        return np.random.choice(self.trajectories,size,replace=False)



class Trajectory:
    def __init__(self):
        self.states = list()
        self.actions = list()
        self.probs = list()
        self.values = list()
        self.rewards = list()
        self.dones = list()
        self.counter = 0

    def add_experience(self,state,action,prob,value,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.counter += 1

    