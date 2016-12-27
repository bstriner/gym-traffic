from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
import traci
import traci.constants as tc
from scipy.misc import imread
from gym import spaces
from string import Template
import os, sys
import numpy as np
import math

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class TrafficEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, lights, netfile, routefile, guifile, addfile, loops=[], tmpfile="tmp.rou.xml",
                 pngfile="tmp.png", mode="gui", detector="detector0", simulation_end=3600):
        # "--end", str(simulation_end),
        self.simulation_end = simulation_end
        self._seed()
        self.loops = loops
        self.loop_variables = [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_TIME_SINCE_DETECTION, tc.LAST_STEP_VEHICLE_NUMBER]
        self.detector = detector
        args = ["--net-file", netfile, "--route-files", tmpfile, "--gui-settings-file",
                guifile, "--additional-files", addfile]
        if mode == "gui":
            binary = "sumo-gui"
            args += ["-S", "-Q"]
        else:
            binary = "sumo"

        with open(routefile) as f:
            self.route = f.read()
        self.tmpfile = tmpfile
        self.pngfile = pngfile
        self.sumo_cmd = [binary] + args
        self.sumo_step = 0
        self.lights = lights
        self.action_space = spaces.Tuple([spaces.Discrete(len(light.actions)) for light in self.lights])
        self.observation_space = spaces.Box(low=float('-inf'), high=float('inf'),
                                            shape=(len(self.loops) * len(self.loop_variables),))
        self.sumo_running = False
        self.viewer = None

    def relative_path(self, *paths):
        os.path.join(os.path.dirname(__file__), *paths)

    def write_routes(self):
        self.route_info = self.route_sample()
        with open(self.tmpfile, 'w') as f:
            f.write(Template(self.route).substitute(self.route_info))

    def _seed(self, seed=None):
        print "Seeding"
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_sumo(self):
        if not self.sumo_running:
            self.write_routes()
            traci.start(self.sumo_cmd)
            for loopid in self.loops:
                traci.inductionloop.subscribe(loopid, self.loop_variables)
            self.sumo_step = 0
            self.sumo_running = True

    def stop_sumo(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    def _reward(self):
        speed = traci.multientryexit.getLastStepMeanSpeed(self.detector)
        return speed

    def _step(self, action):
        self.start_sumo()
        self.sumo_step += 1
        assert (len(action) == len(self.lights))
        for act, light in zip(action, self.lights):
            signal = light.act(act)
            traci.trafficlights.setRedYellowGreenState(light.id, signal)
        traci.simulationStep()
        observation = self._observation()
        reward = self._reward()
        done = self.sumo_step > self.simulation_end
        return observation, reward, done, self.route_info

    def _observation(self):
        res = traci.inductionloop.getSubscriptionResults()
        obs = []
        for loop in self.loops:
            for var in self.loop_variables:
                obs.append(res[loop][var])
        return np.array(obs)

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
        traci.gui.screenshot("View #0", self.pngfile)
        img = imread(self.pngfile, mode="RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
