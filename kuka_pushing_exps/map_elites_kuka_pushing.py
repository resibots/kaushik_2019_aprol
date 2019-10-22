#Insert packages path to sys.path
#So that the codes can be run from any directory
from pathlib import Path
import os 
import sys
this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(this_file_path).parent))
from environments.kuka_pusher_env import *
import numpy as np
import math
import utils.map_elites as EvoAlg
import argparse

env_params = {
    "controlStep": 0.01, 
    "simStep": 0.004, 
    "runtime": 2.0, 
    "jointControlMode": "position", 
    "lateral_friction": 1.0, 
    "toy": 0,
    "object_radius":0.5
}

param_dim = 2
min_inputs = [0.0 for i in range(param_dim)]
max_inputs = [1.0 for i in range(param_dim)]
max_dispalcement = 1.5 #Normalizing parameter for robots displacement

ea_params ={
    "n_gen": 3000,
    "n_niches": 2000,
    "archive_path": './',
    "n_maps": 3,
    "cvt_samples": 25000,
    "batch_size": 100,
    "random_init": 500,
    "sigma_iso": 0.01,
    "sigma_line": 0.2,
    "dump_period": 100,
    "parallel": True,
    "cvt_use_cache": True,
    "min": min_inputs,
    "max": max_inputs,
    "sampled_mutation": False
}

#optional arguments
parser = argparse.ArgumentParser()

parser.add_argument("--lateral_friction", 
                    help='Table_friction', 
                    type=float)

parser.add_argument("--toy", 
                    help="Object id 0-5", 
                    type= int)

arguments = parser.parse_args()
if arguments.lateral_friction is not None: env_params['lateral_friction'] = arguments.lateral_friction
if arguments.toy is not None: env_params['toy'] = arguments.toy

class EnvClass:
    def __init__(self, gui=False, controlStep=env_params["controlStep"], simStep = env_params["simStep"], jointControlMode = env_params["jointControlMode"], lateral_friction=env_params["lateral_friction"], toy=env_params["toy"]):
        self.__actions = []
        self.__controller = KukaController_line()
        if os.getenv('RESIBOTS_DIR') is not None:
            self.__env = Kuka_pusher_env(gui=gui, controlStep=controlStep, simStep = simStep, jointControlMode=jointControlMode, lateral_friction=lateral_friction, toy=toy)
        else:
            self.__env = Kuka_pusher_env(gui=gui, controlStep=controlStep, simStep = simStep, jointControlMode=jointControlMode, lateral_friction=lateral_friction, toy=toy, 
                                        boturdf="/nfs/hal01/rkaushik/projects/multi_dex_python/environments/URDF/kuka_iiwa/kuka_pusher.urdf", 
                                        floorurdf="/nfs/hal01/rkaushik/projects/multi_dex_python/environments/URDF/plane.urdf",
                                        tableurdf = "/nfs/hal01/rkaushik/projects/multi_dex_python/environments/URDF/table_square/table_square.urdf")            
        self.__lock=False
        self.__env.setController(self.__controller)
        self.__gui = gui 
        self.__controlStep = controlStep
        self.__simStep = simStep
 
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
        self.__controller.setParams(params)
    
    def getParams(self):
        return self.__params
    
    def run(self, runtime=3):
        self.__env.run(runtime)   

    def reset(self):
        self.__env.reset()

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
        dim_x = 2
        dim = 2
        fit = data[:, 0:1]
        desc = data[:,1: dim+1]
        trajectories = data[:,dim+1:dim+1+dim_x]
        all_tarjectories += trajectories.tolist()
        all_fit += fit.tolist()
        all_desc += desc.tolist()

    return np.array(all_tarjectories), np.array(all_fit), np.array(all_desc)

Envs = generate_envs(1)

def eval(params):
    for env in Envs:
        if not env.isLock():
            env.lock()
            env.reset()            
            env.setParams(params)

            xy_old = np.array(env.getEnv().get_object_position())[0:2]
            env.run(env_params["runtime"])
            xy_new = np.array(env.getEnv().get_object_position())[0:2]
            
            behavior = (xy_new-xy_old)/max_dispalcement #Assuming max distance to be 1.5
            behavior = (behavior + 1.0)*0.5
            performance = 1.0 #this should be small. Bot to face the direction
            # if env.flipped() or env.isBaseCollision() or env.isHeightExceed():
            #     performance = -10002
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
           
