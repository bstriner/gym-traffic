from gym.envs.registration import register

register(
    id='Traffic-Simple-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    timestep_limit=1000,
)