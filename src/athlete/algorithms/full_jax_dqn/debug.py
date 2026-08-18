import gymnasium as gym
from athlete.algorithms import full_jax_dqn

import athlete

from tqdm import tqdm
import time

USE_NEW_AGENT = True
TRAIN_STEPS = 50_000
EVALUATION_EPISODES = 1


def main():
    env = gym.make("CartPole-v1")

    start_setup_time = time.time()

    if USE_NEW_AGENT:
        agent = full_jax_dqn.make(
            observation_space=env.observation_space, action_space=env.action_space
        )
    else:
        agent = athlete.make(
            algorithm_id="jax_dqn",
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    setup_time = time.time() - start_setup_time

    start_train_time = time.time()
    train(agent, env, train_steps=TRAIN_STEPS)

    train_time = time.time() - start_train_time
    total_time = time.time() - start_setup_time
    print(f"Setup time: {setup_time:.2f} seconds")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")

    # Evaluation
    evaluation_environment = gym.make("CartPole-v1", render_mode="human")
    if USE_NEW_AGENT:
        evaluation_agent = agent.make_evaluation_agent()
    else:
        agent.eval()
        evaluation_agent = agent

    evaluate(
        evaluation_agent,
        evaluation_environment,
        evaluation_episodes=EVALUATION_EPISODES,
    )

    agent_save_path = "jax_dqn_agent.pkl"
    agent.save(save_path=agent_save_path)

    del agent
    del evaluation_agent

    agent = full_jax_dqn.load_agent(save_path=agent_save_path)

    # train(agent, env, train_steps=TRAIN_STEPS)

    if USE_NEW_AGENT:
        evaluation_agent = agent.make_evaluation_agent()
    else:
        agent.eval()
        evaluation_agent = agent

    evaluate(
        evaluation_agent,
        evaluation_environment,
        evaluation_episodes=EVALUATION_EPISODES,
    )


def train(agent, environment, train_steps=100_000):
    observation, env_info = environment.reset()
    action, agent_info = agent.reset_step(observation=observation)

    episode_return = 0
    for step in tqdm(range(train_steps)):
        observation, reward, terminated, truncated, env_info = environment.step(action)
        episode_return += reward
        action, agent_info = agent.step(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            observation, env_info = environment.reset()
            action, agent_info = agent.reset_step(observation=observation)
            # print(f"Episode return: {episode_return}")
            episode_return = 0


def evaluate(evaluation_agent, evaluation_environment, evaluation_episodes=3):
    observation, env_info = evaluation_environment.reset()
    action, agent_info = evaluation_agent.reset_step(observation=observation)

    episode_returns = []
    for episode in range(evaluation_episodes):
        episode_return = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, env_info = (
                evaluation_environment.step(action)
            )
            episode_return += reward
            action, agent_info = evaluation_agent.step(observation=observation)

        observation, env_info = evaluation_environment.reset()
        action, agent_info = evaluation_agent.reset_step(observation=observation)
        episode_returns.append(episode_return)
    print(
        f"Average evaluation return: {sum(episode_returns) / len(episode_returns):.2f}"
    )


if __name__ == "__main__":
    main()
