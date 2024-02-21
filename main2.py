import numpy as np
from Agent import PPOAgent
import gymnasium as gym
if __name__ == "__main__":
    env = gym.make("CartPole-v1",render_mode="human")#SpaceInvaders-v4
    N=20
    batch_size = 5
    learn_epochs = 4
    lr = 0.0003
    n_games = 300
    score_history = list()
    avg_score = 0
    n_steps = 0
    agent = PPOAgent(n_actions=env.action_space.n,batch_size=batch_size,lr=lr,learn_epochs=learn_epochs,input_dims=env.observation_space.shape[0],condition=False)

    for i in range(n_games):
        observation,_= env.reset()
        done=False
        score =0
        loss = 0
        while not done:
            action,prob,val =agent.act(observation)
            #n_steps += 1
            observation_,reward,done,_,_ = env.step(action)
            env.render()
            score += reward
            agent.remember_exp(observation,action,prob,val,reward,done)
            """
            if n_steps % N ==0:
                loss += agent.learn()
            """    
            observation = observation_
        loss += agent.learn()
        if i == 0:
            best_score = score
        agent.remember_tra()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if best_score < score:
            best_score = score
            agent.save_model()
        print("*************")
        print(f"episode {i} score {score} best_score {best_score} loss {loss}")                
