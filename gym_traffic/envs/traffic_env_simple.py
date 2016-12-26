from traffic_env import TrafficEnv

class TrafficEnvSimple(TrafficEnv):
    def __init__(self, mode="gui"):
        self.actions = ["rrrrGGGGrrrrGGGG","GGGGrrrrGGGGrrrr"]
        super(TrafficEnvSimple).__init__(mode=mode)
