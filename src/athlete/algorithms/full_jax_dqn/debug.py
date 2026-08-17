import gymnasium as gym
from athlete.algorithms import full_jax_dqn

from tqdm import tqdm


def main():
    env = gym.make("CartPole-v1")

    agent = full_jax_dqn.make(
        observation_space=env.observation_space, action_space=env.action_space
    )

    observation, env_info = env.reset()
    action, agent_info = agent.reset_step(observation=observation)

    train_steps = 100_000
    episode_return = 0
    for step in tqdm(range(train_steps)):
        observation, reward, terminated, truncated, env_info = env.step(action)
        episode_return += reward
        action, agent_info = agent.step(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            observation, env_info = env.reset()
            action, agent_info = agent.reset_step(observation=observation)
            print(f"Episode return: {episode_return}")
            episode_return = 0
