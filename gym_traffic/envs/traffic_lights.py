class TrafficLight(object):
    def __init__(self, id, actions):
        self.state = 0
        self.step = 0
        self.id = id
        self.actions = actions

    def signal(self):
        return self.actions[self.state]

    def act(self, action):
        if action != self.state and self.action_allowed(action):
            self.state = action
            self.step = 0
        else:
            self.step += 1
        return self.signal()

    def action_allowed(self, action):
        return True


class TrafficLightTwoWay(TrafficLight):
    def __init__(self, id, yield_time=5):
        super(TrafficLightTwoWay, self).__init__(id=id, actions=["GrGr", "rGrG", "yryr", "ryry"])
        self.yield_time = yield_time

    def action_allowed(self, action):
        if self.state == 0:
            return action == 2
        elif self.state == 1:
            return action == 3
        elif self.state == 2:
            return action == 1 and self.step > self.yield_time
        elif self.state == 3:
            return action == 0 and self.step > self.yield_time
        else:
            raise ValueError("Invalid state {}".format(self.state))
        end
