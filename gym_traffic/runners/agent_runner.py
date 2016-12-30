# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"


from gym_traffic.agents import DQN
from gym_traffic.runners import SimpleRunner
import gym
from gym.wrappers import Monitor
import os
from tqdm import tqdm
import pandas as pd


def test(env, agent, path, runner, nb_episodes=5, video_callable=None, force=True):
    monitor = Monitor(env, path, video_callable=video_callable, force=True)
    # tmp, agent.epsilon = agent.epsilon, 0.0
    rewards, losses = runner.run(monitor, agent, nb_episodes=nb_episodes, render=False, train=False)
    # agent.epsilon = tmp
    monitor.close()
    df = pd.DataFrame({"rewards": rewards})
    df.to_csv("{}-test.csv".format(path),index_label="episode")


def run_agent(runner, agent, test_agent, train_env, test_env_func, nb_epoch, nb_episodes, path, video_callable=None,
              test_nb_episodes=10):
    for epoch in tqdm(range(nb_epoch), desc="Training"):
        print("Epoch: {}".format(epoch))
        test_env = test_env_func()
        epoch_path = os.path.join(path, "epoch-{:03d}".format(epoch))
        test(test_env, test_agent, epoch_path, runner, video_callable=video_callable, nb_episodes=test_nb_episodes)
        rewards, losses = runner.run(train_env, agent, nb_episodes=nb_episodes, render=False, train=True)
        test_agent.save("{}.h5".format(epoch_path))
        df = pd.DataFrame({"rewards": rewards, "losses": losses})
        df.to_csv("{}.csv".format(epoch_path),index_label="episode")
