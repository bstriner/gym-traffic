from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
import traci
from scipy.misc import imread
from gym import spaces

import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class TrafficEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mode="gui"):
        self.cfg = r"D:\Projects\gym-traffic\test\sumo_files\config\rlTest.sumocfg"
        if mode == "gui":
            self.sumoBinary = "sumo-gui"
            self.sumo_cmd = [self.sumoBinary, "-S", "-Q", "-c", self.cfg]
        else:
            self.sumoBinary = "sumo"
            self.sumo_cmd = [self.sumoBinary, "-c", self.cfg]
        self.sumo_step = 0
        self.action_space = spaces.Discrete(len(self.actions))
        self.sumo_running = False
        self.viewer = None

    def relative_path(self, *paths):
        os.path.join(os.path.dirname(__file__), *paths)

    def start_sumo(self):
        if not self.sumo_running:
            traci.start(self.sumo_cmd)
            self.sumo_step = 0
            self.sumo_running = True

    def stop_sumo(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    def _step(self, action):
        self.start_sumo()
        self.sumo_step += 1
        traci.trafficlights.setRedYellowGreenState("(2,2)", self.actions[action])
        traci.simulationStep()
        observation = self._observation()
        reward = 0
        done = self.sumo_step > 1000
        info = {}
        return observation, reward, done, info

    def _observation(self):
        return []

    def _reset(self):
        self.stop_sumo()
        self.start_sumo()
        observation = self._observation()
        return observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        filename = "tmp.png"
        traci.gui.screenshot("View #0", filename)
        img = imread(filename, mode="RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
