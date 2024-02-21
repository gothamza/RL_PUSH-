import gymnasium as gym
from nes_py.wrappers import JoypadSpace

env = gym.make('ALE/MarioBros-v5',render_mode="human")
#env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = True
for step in range(2500):
    if done:
        state,_ = env.reset()
    state, reward, done, info,_ = env.step(env.action_space.sample())
    env.render()
env.close()