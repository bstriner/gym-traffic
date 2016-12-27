from traffic_env import TrafficEnv
from traffic_lights import TrafficLightTwoWay
import os


class TrafficEnvSimple(TrafficEnv):
    def __init__(self, mode="gui"):
        lights = [TrafficLightTwoWay(id="0", yield_time=5)]
        loops = ["loop{}".format(i) for i in range(8)]
        basepath = os.path.join(os.path.dirname(__file__), "config", "simple")
        netfile = os.path.join(basepath, "traffic.net.xml")
        routefile = os.path.join(basepath, "traffic.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "traffic.add.xml")
        super(TrafficEnvSimple, self).__init__(mode=mode, lights=lights, netfile=netfile, routefile=routefile,
                                               guifile=guifile, loops=loops, addfile=addfile, simulation_end=600)

    def route_sample(self):
        low = 0.01
        high = 0.1
        return {"ns": self.np_random.uniform(low, high),
                "sn": self.np_random.uniform(low, high),
                "ew": self.np_random.uniform(low, high),
                "we": self.np_random.uniform(low, high)
                }
