# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"


from gym_traffic.agents import DQN, EpsilonExplorer
from gym_traffic.runners import SimpleRunner
import gym
from gym_traffic.runners.agent_runner import run_agent
import sys
import argparse

def build_agent(env):
    return DQN(env.observation_space, env.action_space, memory_size=50, replay_size=32)

def example(gui):
    train_env = gym.make('Traffic-Simple-cli-v0')
    agent = build_agent(train_env)
    path = "output/traffic/simple/dqn"
    explorer = EpsilonExplorer(agent, epsilon=0.5, decay=5e-7)

    if gui:
        def test_env_func():
            return gym.make('Traffic-Simple-gui-v0')
    else:
        def test_env_func():
            return train_env

    runner = SimpleRunner(max_steps_per_episode=1000)
    video_callable = None if gui else False
    run_agent(runner=runner, agent=explorer, test_agent=explorer, train_env=train_env, test_env_func=test_env_func,
              nb_episodes=500, test_nb_episodes=10, nb_epoch=100, path=path, video_callable=video_callable)


def main(argv):
    parser = argparse.ArgumentParser(description='Example DQN implementation of traffic light control.')
    parser.add_argument('-G', '--gui', action="store_true",
                        help='run GUI mode during testing to render videos')

    args = parser.parse_args(argv)
    example(args.gui)


if __name__ == "__main__":
    main(sys.argv[1:])
