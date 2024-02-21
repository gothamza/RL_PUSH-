import gymnasium as gym
import streamlit as st
import numpy as np
from PIL import Image
import time
import os
from DQNAgent import DQNAgent
# User interface for selecting game and model
selected_game = st.selectbox("Select Gym Game", ["CartPole-v1", "SpaceInvaders-v5","Pong-v5","BeamRider-v5","Breakout-v5","Enduro-v5","Qbert-v5","Seaquest-v5"])
selected_model = st.selectbox("Select Model", ["DQN", "ModelB"])


# Button to start the simulation
start_simulation = st.button("Start Simulation")

if start_simulation:
    
    # Load the CartPole environment
    env = gym.make(f"ALE/{selected_game}", render_mode='rgb_array')
    if selected_model == "DQN":
        agent = DQNAgent(env.observation_space.shape,env.action_space.n,lr=0.001,epsilon=0)
    # Reset the environment to get the initial state
    state,_ = env.reset()

    # Define the Streamlit app
    st.title("CartPole Simulation")

    # Directory to save frames as images
    frames_dir = "frames"
    st.markdown(f"Frames will be saved in the directory: `{frames_dir}`")

    # Create the frames directory if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    # Run the simulation
    frame_count = 0
    image_placeholder = st.empty()

    while True:
        # Select an action (for simplicity, always choose action 0)
        action = agent.act(state)

        # Take a step in the environment
        result = env.step(action)
        next_state, reward, done, truncated, info = result
        agent.remember(state,action,reward,next_state,done)
        agent.learn()
        # Render the current state as an image
        frame = env.render()

        # Save the frame as an image
        image = Image.fromarray(frame)
        

        # Display the image in Streamlit
        image_placeholder.image(image, caption=f"Action: {action}", use_column_width=True)

        # Add a small delay to display each image
        time.sleep(0.01)

        frame_count += 1

        # Check if the episode is done
        if done:
            st.text("Episode finished")
            break

    # Display saved frames in Streamlit
    #st.image([image_path for image_path in [f"{frames_dir}/frame_{i:04d}.png" for i in range(frame_count)]], caption="Game frames", use_column_width=True)

    # Close the environment after finishing
    env.close()
