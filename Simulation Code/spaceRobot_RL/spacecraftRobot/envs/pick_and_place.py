import os
from gym import utils
# from gym.envs.robotics import fetch_env
from .spaceRobot_FetchEnv import SpaceRobotFetchEnv

# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = os.path.join("spaceRobot.xml")
MODEL_XML_PATH = os.path.join("spaceRobotFetch.xml")
# MODEL_XML_PATH = os.path.join("spaceRobotFetch_copy.xml")


class SpaceRobotPickAndPlaceEnv(SpaceRobotFetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):

        initial_qpos = {
            'spacecraft_BasefloatingJoint': [0., 0., 5., 1., 0., 0., 1],
            'spacecraft_joint1': 0,
            'spacecraft_joint2': 0,
            'spacecraft_joint3': 0,
            'spacecraft_joint4': 0,
            'spacecraft_joint5': 0,
            'spacecraft_joint6': 0,
            'spacecraft_joint7': 0,
            'object0:joint': [-2.5, 0.5, 5.5, 1., 0., 0., 0.],
        }
        print(initial_qpos, '#######################')
        SpaceRobotFetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)