# This Code is forked from following libraries:
#https://github.com/aravindr93/trajopt
#https://github.com/aravindr93/mjrl/tree/master/setup
# =======================================
# import necessary codes and libraries
from mppi import MPPI
from tqdm import tqdm
import time as timer
import numpy as np
import pickle
import argparse
import json
import os
from xml.etree import ElementTree as ET
from spacerobot_v0 import SpaceRobotEnvV0
from spacerobot_v1 import SpaceRobotEnvV1
# =======================================
# Get command line arguments
parser = argparse.ArgumentParser(description='Trajectory Optimization with filtered MPPI')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to job data with exp params')
args = parser.parse_args()
OUT_DIR = args.output
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
with open(args.config, 'r') as f:
    job_data = eval(f.read())

# Unpack args and make files for easy access
mission = job_data['mission']
PICKLE_FILE = OUT_DIR + '/trajectories' + str(mission) + '.pickle'
EXP_FILE = OUT_DIR + '/job_data' + str(mission) + '.json'
SEED = job_data['seed']
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)
if 'visualize' in job_data.keys():
    VIZ = job_data['visualize']
else:
    VIZ =False

# helper function for visualization
def trigger_tqdm(inp, viz=False):
    if viz:
        return tqdm(inp)
    else:
        return inp
def save_xml(env):
        path = os.getcwd() + '\spaceRobot4mbrl_p.xml'
        print(path)
        b = env.model.get_xml()
        tree = ET.XML(b) 
        with open(path, "wb") as f: 
            f.write(ET.tostring(tree))

# =======================================
# Environment
e = SpaceRobotEnvV0()
e_p = SpaceRobotEnvV1()
# =======================================
# On-board model initialization
if mission==1:
    e_p.model.dof_damping[:] = [ 0.,  0.,  0.,  0.,  0.,  0., 200, 300, 400, 500, 600, 700, 800]
    e_p.model.actuator_ctrlrange[8] =  np.array([-3, 3])
    save_xml(e_p)
elif mission==2:
    e_p.model.dof_damping[:] = [ 0.,  0.,  0.,  0.,  0.,  0., 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    e_p.model.actuator_ctrlrange[8] =  np.array([-3, 3])
    save_xml(e_p)
elif mission==3:
    e_p.model.dof_damping[:] = [ 0.,  0.,  0.,  0.,  0.,  0., 200, 300, 400, 500, 600, 700, 800]
    e_p.model.actuator_ctrlrange[8] =  np.array([0.0, 0.0])
    save_xml(e_p)
elif mission==4:
    e_p.model.dof_damping[:] = [ 0.,  0.,  0.,  0.,  0.,  0., 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    e_p.model.actuator_ctrlrange[8] =  np.array([0.0, 0.0])
    save_xml(e_p)
elif mission=='Uncertain':
    e_p.model.dof_damping[:] = [ 0.,  0.,  0.,  0.,  0.,  0., 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    e_p.model.actuator_ctrlrange[8] =  np.array([-3, 3])
# =======================================
mean = np.zeros(e.action_dim)
sigma = 1.0*np.ones(e.action_dim)
filter_coefs = [sigma, job_data['filter']['beta_0'], job_data['filter']['beta_1'], job_data['filter']['beta_2']]
trajectories = []
ts=timer.time()
for i in range(job_data['num_traj']):
    start_time = timer.time()
    print("Currently optimizing trajectory : %i" % i)
    seed = job_data['seed'] + i*12345
    e.reset()
     
    agent = MPPI(e, e_p, mission,
                 H=job_data['plan_horizon'],
                 paths_per_cpu=job_data['paths_per_cpu'],
                 num_cpu=job_data['num_cpu'],
                 kappa=job_data['kappa'],
                 gamma=job_data['gamma'],
                 mean=mean,
                 filter_coefs=filter_coefs,
                 default_act=job_data['default_act'],
                 seed=seed,
                 )
    
    for t in trigger_tqdm(range(job_data['H_total']), VIZ):
        agent.train_step(job_data['num_iter'])
    
    end_time = timer.time()
    print("Trajectory reward = %f" % np.sum(agent.sol_reward))
    print("Optimization time for this trajectory = %f" % (end_time - start_time))
    trajectories.append(agent)
    pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))

print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))

if VIZ:
    _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
    for i in range(3):
        [traj.animate_result() for traj in trajectories]

# =======================================
