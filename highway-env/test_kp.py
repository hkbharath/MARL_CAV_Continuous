import highway_env
import gym


env = gym.make("merge-multi-agent-v0")
for _ in range(1):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        