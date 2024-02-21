
import gymnasium as gym
import streamlit as st
import numpy as np
from PIL import Image
import time
import os
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt

# Function to run training and display results
def run_training(selected_game, selected_model, epochs):
    losses = []
    rewards = []

    # Load the selected game environment
    env = gym.make(f"ALE/{selected_game}", render_mode='rgb_array')

    # Initialize the DQNAgent
    if selected_model == "DQN":
        agent = DQNAgent(env.observation_space.shape, env.action_space.n, lr=0.001, epsilon=0,gama=0.99,memory_size=1000,batch_size = 32,epsilon=1,chkpt_file="DQN.pth")

    # Reset the environment to get the initial state
    state, _ = env.reset()

    # Directory to save frames as images
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    image_placeholder = st.empty()
    # Run the training loop
    for epoch in range(1, epochs + 1):
        done = False
        total_reward = 0.0
        state, _ = env.reset()
        loss_t = 0.0
        reward_t = 0.0
        lifes = env.unwrapped.ale.lives()

        while not done:
            action = agent.act(state)
            state_, reward, done, _, _ = env.step(action)
            current_lifes = env.unwrapped.ale.lives()

            # Punish for dying
            if current_lifes < lifes:
                lifes = current_lifes
                reward_t -= 50

            # Actual reward
            reward_t += reward
            # For time
            reward_t -= 1

            agent.remember(state, action, reward, state_, done)
            loss_t += agent.learn()

            # Render the current state as an image
            frame = env.render()

            # Save the frame as an image
            image = Image.fromarray(frame)
                    # Display the image in Streamlit
            image_placeholder.image(image, caption=f"Action: {action}", use_column_width=True)

            # Add a small delay to display each image
            time.sleep(0.01)

        agent.be_reasonable(epoch)
        losses.append(loss_t)
        rewards.append(reward_t)

        if (epoch % 100) == 0:
            print(f"--------- saving the model at epoch {epoch} ---------")
            agent.save()

        print(f"******** loss at epoch {epoch} = {loss_t} ********")
        print(f"******** reward at epoch {epoch} = {reward_t} ********")

    return losses, rewards

# Streamlit app
st.title("Reinforcement Learning Training")

# User interface for selecting game, model, and epochs
selected_game = st.selectbox("Select Gym Game", ["SpaceInvaders-v5", "CartPole-v5", "Pong-v5", "BeamRider-v5", "Breakout-v5", "Enduro-v5", "Qbert-v5", "Seaquest-v5"])
selected_model = st.selectbox("Select Model", ["DQN", "ModelB"])
epochs = st.slider("Number of Epochs", min_value=1, max_value=500, value=100, step=1)

# Button to start the simulation
start_simulation = st.button("Start Training")

if start_simulation:
    # Run training and get results
    losses, rewards = run_training(selected_game, selected_model, epochs)

    # Display the training plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Plot on the left subplot
    axes[0].plot(range(epochs), losses, label='loss')
    axes[0].set_title('Loss per Epoch')
    axes[0].legend()

    # Plot on the right subplot
    axes[1].plot(range(epochs), rewards, label='reward', color='orange')
    axes[1].set_title('Reward per Epoch')
    axes[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots in Streamlit
    st.pyplot(fig)

    # Optionally, you can display other information or save the plots as images.

    # Display saved frames in Streamlit
    #frame_images = [Image.open(f"{frames_dir}/frame_{i:04d}.png") for i in range(1, epochs + 1)]
    #st.image(frame_images, caption="Game frames", use_column_width=True)

    # Close the environment after finishing
    env.close()
