from gym.envs.registration import register


def _merge(a, b):
    a.update(b)
    return a


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

register(
    id='SpaceRobot-v0',
    entry_point='spacecraftRobot.envs:SpaceRobotEnv',
    max_episode_steps=75,
)

register(
    id='SpaceRobotContinuous-v0',
    entry_point='spacecraftRobot.envs:SpaceRobotContinuousEnv',
    max_episode_steps=200,
)

register(
        id='SpaceRobotPickAndPlace-v0',
        entry_point='spacecraftRobot.envs:SpaceRobotPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=100,
)

register(
        id='SpaceRobotReach-v0',
        entry_point='spacecraftRobot.envs:SpaceRobotReachEnv',
        kwargs=kwargs,
        max_episode_steps=100,
)