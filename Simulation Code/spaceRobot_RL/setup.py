import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='spacecraftRobot',
      version='0.0.1',
      packages=find_packages(),
      author='Ash Babu',
      url='https://github.com/ashbabu/spaceRobot_RL.git',
      long_description=read('README.md'),
      keywords='machine learning reinforcement learning MuJoCo',
      install_requires=['gym',
                        'numpy',
                        'mujoco-py',
                        'glfw']  # And any other dependencies required
)