from gym_traffic.runners import SimpleRunner
import gym
from gym_traffic.runners.agent_runner import test
import sys
from example_gym_traffic_dqn import build_agent


def render_example(input_path, output_path):
    env = gym.make('Traffic-Simple-gui-v0')
    agent = build_agent(env)
    agent.load(input_path)
    runner = SimpleRunner(max_steps_per_episode=1000)
    test(env, agent, output_path, runner, nb_episodes=50, video_callable=None, force=True)


def main():
    render_example("output/epoch-049.h5", "output/epoch-049")
    render_example("output/epoch-000.h5", "output/epoch-000")


if __name__ == "__main__":
    main()
