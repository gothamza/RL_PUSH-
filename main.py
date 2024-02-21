

from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
    env.metadata["render_fps"] = 240

    epochs = 100
    losses = list()
    rewards = list()
    agent = DQNAgent(env.observation_space.shape,env.action_space.n,lr=0.001)


    for epoch in range(1,epochs+1):
        done = False
        total_reward = 0.0
        state,_ = env.reset()
        loss_t = 0.0
        reward_t = 0.0
        lifes = env.unwrapped.ale.lives()
        print(f"epsilon : {agent.epsilon}")
        while not done :
            
            action = agent.act(state)
            state_,reward,done,_,_ = env.step(action)
            current_lifes = env.unwrapped.ale.lives()
            #punish for dying
            if current_lifes < lifes:
                lifes = current_lifes
                reward_t -= 50
            #actual reward
            reward_t += reward 
            #for time
            reward_t -= 1
            env.render()
            agent.remember(state,action,reward,state_,done)
            loss_t += agent.learn()
            
        agent.be_reasonable(epoch)
        losses.append(loss_t)
        rewards.append(reward_t)   
        
        if (epoch % 100) ==0 :
            print(f"--------- saving the model ---------")
            agent.save()
        
        print(f"******** loss at epoch {epoch} = {loss_t} ********")
        print(f"******** reward at epoch {epoch} = {reward_t} ********")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Plot on the left subplot
    axes[0].plot(range(epochs), losses, label='loss')
    axes[0].set_title('loss per epoch')
    axes[0].legend()

    # Plot on the right subplot
    axes[1].plot(range(epoch), rewards, label='reward', color='orange')
    axes[1].set_title('reward per epoch')
    axes[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()