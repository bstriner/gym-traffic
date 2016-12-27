from gym.envs.registration import register

register(
    id='Traffic-Simple-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    timestep_limit=10000,
    kwargs={"mode":"gui"},
    nondeterministic=True
)

register(
    id='Traffic-Simple-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    timestep_limit=10000,
    kwargs={"mode":"cli"},
    nondeterministic=True
)
