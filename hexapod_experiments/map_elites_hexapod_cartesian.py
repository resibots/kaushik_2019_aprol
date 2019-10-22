#Insert packages path to sys.path
#So that the codes can be run from any directory
from pathlib import Path
import os 
import sys
this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(this_file_path).parent))
from environments.hexapod_env_v2 import *
import numpy as np
import math
import utils.map_elites as EvoAlg
import os
import argparse

env_params = {
    "controlStep": 0.01, 
    "simStep": 0.004, 
    "runtime": 3.0, 
    "jointControlMode": "velocity", 
    "lateral_friction": 1.0, 
    "blocked_legs": []
}

param_dim = 36
min_inputs = [0.0 for i in range(param_dim)]
max_inputs = [1.0 for i in range(param_dim)]
max_dispalcement = 1.5 #Normalizing parameter for robots displacement

ea_params ={
    "n_gen": 5000,
    "n_niches": 2000,
    "archive_path": './',
    "n_maps": 3,
    "cvt_samples": 25000,
    "batch_size": 100,
    "random_init": 3000,
    "sigma_iso": 0.01,
    "sigma_line": 0.2,
    "dump_period": 1000,
    "parallel": True,
    "cvt_use_cache": True,
    "min": min_inputs,
    "max": max_inputs,
    "sampled_mutation_rate": 0.05,
    "sampled_mutation": True,
    "sampled_mutation_array": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, \
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, \
    0.90, 0.95, 1.0]
}

#optional arguments
parser = argparse.ArgumentParser()

parser.add_argument("--lateral_friction", 
                    help='BO parameter', 
                    type=float)

parser.add_argument("--blocked_legs", 
                    help="Inddices of the legs to be blocked", 
                    type= int,
                    nargs='+')

arguments = parser.parse_args()
if arguments.lateral_friction is not None: env_params['lateral_friction'] = arguments.lateral_friction
if arguments.blocked_legs is not None: env_params['blocked_legs'] = arguments.blocked_legs

class EnvClass:
    def __init__(self, gui=False, controlStep=env_params["controlStep"], simStep = env_params["simStep"], runtime=env_params["runtime"], jointControlMode = env_params["jointControlMode"], lateral_friction=env_params["lateral_friction"], blocked_legs=env_params["blocked_legs"]):
        self.__actions = []
        self.__controller = HexaController()
        if os.getenv('RESIBOTS_DIR') is not None:
            self.__env = Hexapod_env(gui=gui, controlStep=controlStep, simStep = simStep, jointControlMode=jointControlMode, lateral_friction=lateral_friction)
        else:
            self.__env = Hexapod_env(gui=gui, controlStep=controlStep, simStep = simStep, jointControlMode=jointControlMode, lateral_friction=lateral_friction, boturdf="/nfs/hal01/rkaushik/projects/multi_dex_python/environments/URDF/pexod.urdf", floorurdf="/nfs/hal01/rkaushik/projects/multi_dex_python/environments/URDF/plane.urdf")            
        self.__lock=False
        self.__env.setController(self.__controller)
        self.__gui = gui 
        self.__controlStep = controlStep
        self.__simStep = simStep
        self.__runtime = runtime
        self.__blocked_legs = blocked_legs
        self.__block_mask = np.ones(36)
        
        for leg in self.__blocked_legs:
            assert leg < 6 and leg >= 0
            # disable first joint
            self.__block_mask[6*leg] = 0   
            self.__block_mask[6*leg+1] = 0   
            self.__block_mask[6*leg+2] = 0  
            
            # disable 2nd joint
            self.__block_mask[6*leg+3] = 0   
            self.__block_mask[6*leg+4] = 0   
            self.__block_mask[6*leg+5] = 0
 
    def lock(self):
        self.__lock=True
    
    def release(self):
        self.__lock=False
    
    def isLock(self):
        return self.__lock

    def getEnv(self):
        return self.__env
    
    def setParams(self, params):
        self.__params = params
        self.__controller.setParams(params * self.__block_mask)
    
    def getParams(self):
        return self.__params
    
    def getFinalCM(self):
        return self.__env.getState()[0:2]
    
    def flipped(self):
        return self.__env.flipped()
    
    def isBaseCollision(self):
        return self.__env.isBaseCollision()         
    
    def isHeightExceed(self):
        return self.__env.isHeightExceed()
    
    def run(self, runtime=None):
        if runtime is None:
            runtime = self.__runtime
        self.__env.run(runtime)   

    def reset(self):
        self.__env.reset()
    
    def polar_coordinate(self):
        '''
        Returns final polar coordinate of the robot
        range: [0,-180] to [infinity, 180] 
        '''
        final_state = self.__env.getState()[0:2]
        unit_x = [1. , 0.]
        r = np.linalg.norm(final_state)
        costheta = np.dot(unit_x, final_state)/ (np.linalg.norm(unit_x) * np.linalg.norm(final_state)) #cos(theta) = A.B/mod(A)*mod(B)
        if final_state[1] is not 0.0:
            theta = np.rad2deg(np.arccos(costheta)) * (final_state[1]/np.abs(final_state[1]))
        else:
            theta = np.rad2deg(np.arccos(costheta))
        return np.array([r, theta])
    
    def getFinalBodyAngle(self):
        final_face_vec = [self.getEnv().getRotation()[0], self.getEnv().getRotation()[3]] # first two element of 1st column of rotation matrix
        final_tail_vec = -1.0 * np.array(final_face_vec)
        final_position_vec = self.__env.getState()[0:2]
        if final_position_vec[0] > 0.0: #moves in +x direction
            bodyVector = final_face_vec
        else:
            bodyVector = final_tail_vec            
        costheta = np.dot(bodyVector, final_position_vec)/ (np.linalg.norm(bodyVector) * np.linalg.norm(final_position_vec))
        theta = np.rad2deg(np.arccos(costheta)) #absolute faceangle with the position vector [0,180]
        return theta 
    
    def cartesian_coordinate(self):
        return np.array(self.__env.getState()[0:2])

def generate_envs(n_envs):
    Envs = []
    for i in range(n_envs):
        Envs.append(EnvClass())
    return Envs

def load_map(files):
    all_tarjectories = []
    all_fit = []
    all_desc = []
    for f in files:
        data = np.loadtxt(f)
        dim_x = 36
        dim = 2
        fit = data[:, 0:1]
        desc = data[:,1: dim+1]
        trajectories = data[:,dim+1:dim+1+dim_x]
        all_tarjectories += trajectories.tolist()
        all_fit += fit.tolist()
        all_desc += desc.tolist()

    return np.array(all_tarjectories), np.array(all_fit), np.array(all_desc)

Envs = generate_envs(5)

def eval(params):
    for env in Envs:
        if not env.isLock():
            env.lock()
            env.reset()
            env.setParams(36*[0])
            env.run(1.0)
            
            env.setParams(params)
            env.run()
            behavior = env.cartesian_coordinate()/max_dispalcement #Assuming max distance to be 1.5
            behavior = (behavior + 1.0)*0.5
            performance = -env.getFinalBodyAngle() #this should be small. Bot to face the direction
            if env.flipped() or env.isBaseCollision() or env.isHeightExceed():
                performance = -10002
            env.release()
            return performance, behavior #range [0,1] for map elites

def validate_safety(indiv):
    '''
    Gets the object of the species and validates
    whether to insert in the archive.
    '''
    if indiv.fitness < -10000:
        return False
    else:
        return True

with open('env_params.txt', 'w') as file:
    file.write(str(env_params))

with open('ea_params.txt', 'w') as file:
    file.write(str(ea_params))

for i in range(ea_params["n_maps"]):
    init_population = None
    archive_files = []
    new_file_prefix=''
    for k in range(ea_params["n_maps"]+1): 
        final_archive = ea_params["archive_path"] + 'archive_map'+str(k)+'_'+str( ea_params["n_gen"])+'.dat'
        exists = os.path.isfile(final_archive)
        if exists:
            archive_files.append(final_archive)
            print (final_archive)
        else:
            new_file_prefix = 'archive_map'+str(k)+'_'
            break
    
    archive = EvoAlg.compute(
                    dim_map=2, 
                    dim_x=param_dim, 
                    f=eval, 
                    n_niches=ea_params["n_niches"], 
                    n_gen=ea_params["n_gen"], 
                    params=ea_params, 
                    file_prefix=new_file_prefix, 
                    validate_indiv=validate_safety
                )
           
