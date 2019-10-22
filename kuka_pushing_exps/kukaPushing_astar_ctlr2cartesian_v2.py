#Insert packages path to sys.path
#So that the codes can be run from any directory
from pathlib import Path
import os 
import sys
this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(this_file_path).parent))
import numpy as np
import math
from environments.kuka_pusher_env import * 
from GPy.mappings import Constant
import GPy
import copy
import matplotlib.pyplot as plt
import argparse
import glob
import os
from path_planning.a_star_algorithm import A_star 
import json

args = { 
        'directions': 40, # Desired mini_archive directions
        'ucb_const': 0.05, # Exploration vs exploitation of GP model
        'gp_opt_hp': False,
        'archive_prob': 0.0, # probability of selecting from mini_archive
        'toy': 0,
        'gui': False,
        'visualization_speed': 2.0,
        'search_size': 300, #Pick top search_size number of controllers from the archives
        'kernel_l': 0.3,
        'kernel_var': 0.03, #Angle range to search for high performing behavior  
        'lateral_friction': 1.0,
        'archives_path': "./data/kuka_pusher_prior",
        'mapelites_generations': 3000,
        'ablation': None, # "only_ucb", "only_learning",
        'objectEulerAngles': None} #angles in radian

#optional arguments
parser = argparse.ArgumentParser()

parser.add_argument("--archives_path", 
                    help='Path to MAP-elites archives', 
                    type=str)

parser.add_argument("--objectEulerAngles", 
                    help="Euler angles (xyz) in radian", 
                    type= float,
                    nargs='+')

parser.add_argument("--ablation", 
                    help=' "only_ucb" or "only_learning"', 
                    choices=['only_ucb','only_learning'],
                    type=str)

parser.add_argument("--mapelites_generations", 
                    help='MAP-elites generations used. Required to decide the file names to load from', 
                    type=int)

parser.add_argument("--directions", 
                    help='Desired mini archive directions', 
                    type=int)

parser.add_argument("--ucb_const", 
                    help='ucb constant multiplier parameter', 
                    type=float)

parser.add_argument("--toy", 
                    help="Toy to be used for pushing", 
                    type= int)

parser.add_argument("--gui", 
                    help="Whether visualization required", 
                    action= "store_true")

parser.add_argument("--visualization_speed", 
                    help="Speed of visualization", 
                    type=float)

parser.add_argument("--kernel_l", 
                    help="gp exponential kernel length scale", 
                    type=float)

parser.add_argument("--kernel_var", 
                    help="gp exponential kernel variance", 
                    type=float)

parser.add_argument("--search_angle", 
                    help="Angular range around desired direction to search for high performing controller", 
                    type=float)

parser.add_argument("--search_size", 
                    help="Number of controllers to pick from the given angular range for BO", 
                    type=int)

parser.add_argument("--lateral_friction", 
                    help="Floor friction", 
                    type=float)

arguments = parser.parse_args()
if arguments.archives_path is not None: args['archives_path'] = arguments.archives_path.rstrip("/")
if arguments.mapelites_generations is not None: args['mapelites_generations'] = arguments.mapelites_generations
if arguments.directions is not None: args['directions'] = arguments.directions
if arguments.ucb_const is not None: args['ucb_const'] = arguments.ucb_const
if arguments.toy is not None: args['toy'] = arguments.toy
if arguments.gui is not None: args['gui'] = arguments.gui
if arguments.visualization_speed is not None: args['visualization_speed'] = arguments.visualization_speed
if arguments.kernel_l is not None: args['kernel_l'] = arguments.kernel_l
if arguments.kernel_var is not None: args['kernel_var'] = arguments.kernel_var
if arguments.search_size is not None: args['search_size'] = arguments.search_size
if arguments.lateral_friction is not None: args['lateral_friction'] = arguments.lateral_friction
if arguments.ablation is not None: args['ablation'] = arguments.ablation
if arguments.objectEulerAngles is not None: args['objectEulerAngles'] = arguments.objectEulerAngles

if args['objectEulerAngles'] == [-1]:
    args['objectEulerAngles'] = [0,0, np.random.rand()*2*3.1415-3.1415]

sim_time = 2.0
env_real = Kuka_pusher_env(gui=args['gui'], visualizationSpeed=args['visualization_speed'], jointControlMode="position", lateral_friction=args["lateral_friction"], toy=args["toy"], objStartPositon=[0.3, -1.2, 1.5], objEulerAngles=args["objectEulerAngles"])
ctlr = KukaController_line(params = np.array([0.3, 0.8]), object_radius=0.6)
env_real.setController(ctlr)

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

class GP_model():
    
    def __init__(self, inputs, outputs, mean_functions, opt_restarts=3, opt_hyperparams=True, noise_var=1e-3):
        self.dim_in = len(inputs[0])
        self.dim_out = len(outputs[0])
        assert len(mean_functions) == self.dim_out
        assert inputs.ndim == 2
        assert outputs.ndim == 2
        self.m = []
        for i in range(self.dim_out):
            kernel = GPy.kern.RBF(input_dim=len(inputs[0]), variance=args['kernel_var'], lengthscale=args['kernel_l'])
            self.m.append(GPy.models.GPRegression(inputs, outputs[:, i].reshape(-1, 1), kernel=kernel, mean_function=mean_functions[i], normalizer=False, noise_var=noise_var))
            if opt_hyperparams:
                self.m[i].optimize_restarts(num_restarts = opt_restarts)
    
    def predict(self, inputs):
        assert inputs.ndim == 2
        mean = []
        var = []
        for model in self.m:
            mu, sig = model.predict(inputs)
            mean.append(mu.flatten())
            var.append(sig.flatten())
        return np.array(mean).transpose() , np.array(var).transpose()
    
    def getLikelihood(self, inputs, point):
        assert inputs.ndim == 2
        assert point.shape == (self.dim_out,)
        log_likelihoods = []
        for i, model in enumerate(self.m):
            mu, var = model.predict(inputs)
            log_lik = -0.5*np.log(2*np.pi*var) - np.power(mu - point[i], 2)/(2. * var + 1e-10)
            log_likelihoods.append(log_lik.flatten())
        
        total_lik = np.sum(log_likelihoods, axis=0)
        return np.exp(total_lik)

def execute_real_bot(param, runtime=2, visual_goal=[], lateral_friction=None):
    pp = param.copy()
    if lateral_friction is not None:
        env_real.setFriction(lateral_friction)    
    env_real.getController().setParams(pp)
    env_real.run(runtime, goal_position=visual_goal)

    final_state = env_real.get_object_position()[0:2]    
    return np.array(final_state)

def convert_polar2desc(polar):
    r = polar[0]
    theta = polar[1]
    r_desc = (r/5.0) * 2.0 - 1.0
    theta_desc = theta/180.0
    return np.array([r_desc, theta_desc])

def convert_desc2polar(desc):
    r_desc = desc[0]
    theta_desc = desc[1]
    r = (r_desc + 1.0) * 0.5 * 5.0
    theta = theta_desc *180.0
    return np.array([r, theta])

def convert_polar2cart(polar):
    r = polar[0]
    theta = polar[1]
    return np.array([r * np.cos(theta), r * np.sin(theta)])

def convert_cart2polar(cart):
    x = cart[0]
    y = cart[1]
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    return np.array([r,np.rad2deg(theta)])

def getCM():
    return np.array(env_real.get_object_position()[0:2])
    
def relative_coordinate(absolute_point, face_angle, cm):
    #World frame to robots frame
    rm = np.array([[np.cos(np.deg2rad(face_angle)), np.sin(np.deg2rad(face_angle))],
                    [-np.sin(np.deg2rad(face_angle)), np.cos(np.deg2rad(face_angle))]])
    point = absolute_point - cm
    return np.matmul(rm,point)

def add_obstacles(env, obstacles):
    for obstacle in obstacles:
        colBoxId = env_real.p.createCollisionShape(env_real.p.GEOM_BOX,halfExtents=[(obstacle[2]-obstacle[0])/2.0, (obstacle[3]-obstacle[1])/2.0, 0.2],physicsClientId=env_real.physicsClient)
        env_real.p.createMultiBody(baseMass=10, baseCollisionShapeIndex = colBoxId, basePosition = [(obstacle[0]+obstacle[2])/2.0, (obstacle[1]+obstacle[3])/2.0, 0.5],physicsClientId=env_real.physicsClient)

def add_goal(env, goal):
    visualGoalShapeId = env_real.p.createVisualShape(shapeType=env_real.p.GEOM_CYLINDER, radius=0.3, length=0.04, visualFramePosition=[0.,0.,0.], visualFrameOrientation=env_real.p.getQuaternionFromEuler([0,0,0]) , rgbaColor=[1, 0, 1, 0.5], specularColor=[0.5,0.9, 0.9, 1.0], physicsClientId=env_real.physicsClient)    
    goalid = env_real.p.createMultiBody(baseMass=0.0,baseInertialFramePosition=[0,0,0], baseVisualShapeIndex = visualGoalShapeId, basePosition = [goal[0], goal[1], 1.32], useMaximalCoordinates=True, physicsClientId=env_real.physicsClient)
    return goalid

class Custom_mean_x(Constant):
    def __init__(self, input_dim, output_dim, dictionary):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dictionary = dictionary
        super(Custom_mean_x, self).__init__(input_dim=input_dim, output_dim=output_dim)

    def f(self, X):
        return  np.array([self.dictionary[tuple(x)] for x in X])

class Custom_mean_y(Constant):
    def __init__(self, input_dim, output_dim, dictionary):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dictionary = dictionary
        super(Custom_mean_y, self).__init__(input_dim=input_dim, output_dim=output_dim)

    def f(self, X):
        return  np.array([self.dictionary[tuple(x)] for x in X])

def estimate_posteriors_ucb(current_parameter, current_obs, expected_obs, leg_blocks, frictions, prior_map_index, prior_map_trials, prior_map_means):
    ucb = np.zeros(len(prior_map_means))
    t = np.sum(prior_map_trials)
    posterior =  np.zeros(len(prior_map_means))
    #Compute UCBs
    for i in range(len(prior_map_means)):
        ucb[i] = args["ucb_const"] * np.sqrt(2*np.log(t)/prior_map_trials[i])
    
    #Update means
    perf = np.exp(-3.0 * np.linalg.norm(current_obs-expected_obs))
    prior_map_means[prior_map_index] = ((prior_map_trials[prior_map_index]-1)*prior_map_means[prior_map_index] + perf)/prior_map_trials[prior_map_index]

    posterior = (np.array(prior_map_means) + np.array(ucb))/(np.array(prior_map_means) + np.array(ucb)).sum()
    return posterior 

def gaussian_likelihood(means, vars, point):
    log_likelihoods = []
    for i in range(means.shape[1]):
        var = vars.transpose()[i]
        mu = means.transpose()[i]
        log_lik = -0.5*np.log(2*np.pi*var) - np.power(mu - point[i], 2)/(2. * var + 1e-10)
        log_likelihoods.append(log_lik.flatten())
    
    total_lik = np.sum(log_likelihoods, axis=0)
    return np.exp(total_lik)

def main():
    print ("\n\nMake sure GPy Version: 1.9.6 \n **************************")
    maps_dirs = glob.glob(args['archives_path']+"/*")
    assert len(maps_dirs) > 0 , "No archives found. Check the --archive_path option"
    np.random.shuffle(maps_dirs)
    blocks = []
    frictions = []
    map_files = []

    # keep_map = [[0,3], [1,4],[2,5],[2,3],[2,4],[0],[4]]
    # keep_map = [[3]]
    for dd in maps_dirs:
        with open(dd+"/env_params.txt", 'r') as f:
            s = f.readlines()[0]
        json_acceptable_string = s.replace("'", "\"")
        d = json.loads(json_acceptable_string)
        if args["ablation"] == "only_learning":
            #Taking toy map 10 as prior
            if float(d["lateral_friction"])==1.0 and len(blocks)< 1 and d["toy"]==10:
                blocks.append(d["toy"])
                frictions.append(d["lateral_friction"])
                map_files.append(dd+"/archive_map0_" + str(args["mapelites_generations"]) + ".dat")
        else:
            #Taking all toys maps as prior, except the toy itself
            if float(d["lateral_friction"])==1.0 and not d["toy"]==args["toy"]:
                blocks.append(d["toy"])
                frictions.append(d["lateral_friction"])
                map_files.append(dd+"/archive_map0_" + str(args["mapelites_generations"]) + ".dat")
        

    print("MAPs loaded as prior: ", map_files, "\n")
    gp_models = len(map_files)*[object]
    mean_functions = []
    parameters = []
    descriptors = []
    for f in map_files:
        params, _ , desc = load_map([f])
        desc = (desc * 2.0 - 1.0) * 1.5
        parameters.append(params.copy())
        descriptors.append(desc.copy())

    parameters = np.array(parameters)
    descriptors = np.array(descriptors)

    # Model settings
    for prior_index in range(len(map_files)):
        dictionary_x = dict()
        dictionary_y = dict()
        desc = descriptors[prior_index]
        params = parameters[prior_index]
        for i, dd in enumerate(desc):
            dictionary_x[tuple(dd)] = [dd[0]]
            dictionary_y[tuple(dd)] = [dd[1]]
        mf_x = Custom_mean_x(2, 1, dictionary_x)
        mf_y = Custom_mean_y(2, 1, dictionary_y)
        mean_functions.append([mf_x.copy(), mf_y.copy()])
    
    #Environment settings
    Obstacles = np.array([[-30, -20, -50, -10]])
    boundary = [-3, 3, -3, 3]
    all_goals = [[0.4, 0.7], [0.5, -0.8], [0.2, 0.7], [0.2, -0.8]]
    main_goal= np.array([0.4, 0.7])
    # add_goal(env_real, main_goal)
    step = 0

    executed_controllers = []
    executed_descriptors = []
    observations = []

    executed_descriptors_mapwise = [[] for i in range(len(map_files))]
    observations_mapwise = [[] for i in range(len(map_files))]

    prior_map_index = []
    prior_map_trials = np.ones(len(map_files))
    prior_map_means = np.ones(len(map_files))*1.0
    goalid = -1
    for test in range(1):
        env_real.reset()
        bot_directon = env_real.get_obj_orientation()[2]
        bot_cm = getCM()
        first = True
        step = 0

        trajectory = []
        belief_map = []
        belief_action = []
        trajectory.append({"bot_cm":bot_cm, "bot_directon":bot_directon, "step":step, "action": None, "controller": None, "belief_map": None, "belief_action": None, "failed": False})
        
        main_goal = np.array(all_goals.pop(0))
        goalid = add_goal(env_real, main_goal)
        while True:    
            step +=1
            bot_full_orientation = env_real.get_obj_orientation()
            bot_directon = env_real.get_obj_orientation()[2]
            bot_cm = getCM()
            
            #plan path
            planner = A_star(Obstacles, boundary, resolution=0.05)
            path = planner.plan(bot_cm, main_goal)
            # goal = np.array(path[min(6, len(path)-1)])
            goal = np.array(path[min(18, len(path)-1)])
            # goal = np.array(path[5]) if len(path) > 5 else main_goal
            visual_goal = [goal[0], goal[1], 1.32]        
            relative_goal = relative_coordinate(goal, bot_directon, bot_cm)
            pp = []
            
            if first:
                diff = np.power(descriptors[0]-relative_goal,2).sum(axis=1)
                index = np.argmin(diff)
                pp = parameters[0][index]
                prior_map_trials[0] += 1.0
                prior_map_index.append(0)
                executed_descriptors.append(descriptors[prior_map_index[-1]][index])
                executed_descriptors_mapwise[prior_map_index[-1]].append(executed_descriptors[-1])
            else:
                #Get the posterior over the prior maps
                posterior = estimate_posteriors_ucb(executed_controllers[-1], observations[-1], executed_descriptors[-1] ,blocks, frictions, prior_map_index[-1], prior_map_trials, prior_map_means)
                belief_map = {"block_belief": blocks[np.argmax(posterior)] , "friction_belief": frictions[np.argmax(posterior)], "full_map_posterior": posterior}

                
                 #Directly from prior (abletion of full posterior over controllers)
                #--------------------
                if args["ablation"] == "only_ucb": 
                    prior_map_index.append(np.argmax(posterior))
                    diff = np.power(descriptors[np.argmax(posterior)]-relative_goal,2).sum(axis=1)
                    pp = parameters[np.argmax(posterior)][np.argmin(diff)]
                    prior_map_trials[np.argmax(posterior)] += 1.0
                    executed_descriptors.append(descriptors[np.argmax(posterior)][np.argmin(diff)])
                    executed_descriptors_mapwise[prior_map_index[-1]].append(executed_descriptors[-1])
                
                #Get the likelihood of the controllers given priors:
                else:
                    if args["ablation"] == "only_leaning":
                        posterior = posterior * 0.0 + 1.0
                    likelihoods = []
                    test_controllers = []
                    test_descriptors = []
                    for prior_index in range(len(map_files)):
                        diff = np.power(descriptors[prior_index]-relative_goal,2).sum(axis=1)
                        indices = np.argsort(diff)[0:args["search_size"]]
                        cc = parameters[prior_index][indices]
                        dd = descriptors[prior_index][indices]

                        if isinstance(gp_models[prior_index], GP_model):
                            #If leanred model available
                            lik = gp_models[prior_index].getLikelihood(dd, relative_goal) * posterior[prior_index]
                        else:
                            #If model is yet not learned
                            lik = gaussian_likelihood(descriptors[prior_index][indices], np.ones(2*len(indices)).reshape(-1, 2)* args['kernel_var'], relative_goal) * posterior[prior_index]
                            
                        likelihoods.append(lik[np.argmax(lik)])
                        test_controllers.append(cc[np.argmax(lik)])
                        test_descriptors.append(dd[np.argmax(lik)])

                    pp = test_controllers[np.argmax(likelihoods)]
                    dd = test_descriptors[np.argmax(likelihoods)]
                    prior_map_trials[np.argmax(likelihoods)] += 1.0
                    prior_map_index.append(np.argmax(likelihoods))
                    belief_action = {"block_belief": blocks[np.argmax(likelihoods)] , "friction_belief": frictions[np.argmax(likelihoods)], "sorted_actions": test_descriptors, "full_action_posterior": likelihoods}
                    executed_descriptors.append(dd)
                    executed_descriptors_mapwise[prior_map_index[-1]].append(dd)
                
            #Execute bot
            obs = execute_real_bot(pp, 2, visual_goal=visual_goal)
            relative_obs = relative_coordinate(obs, bot_directon, bot_cm)

            #Learn model
            executed_controllers.append(pp)
            observations.append(relative_obs)
            observations_mapwise[prior_map_index[-1]].append(relative_obs)
            gp_models[prior_map_index[-1]] = GP_model(np.array(executed_descriptors_mapwise[prior_map_index[-1]]), 
                                                    np.array(observations_mapwise[prior_map_index[-1]]), 
                                                    mean_functions = mean_functions[prior_map_index[-1]], 
                                                    opt_hyperparams=False)
            first = False
            
            trajectory.append({
                            "bot_cm":getCM(), 
                            "bot_directon":env_real.get_obj_orientation()[2], 
                            "step":step, 
                            "action": executed_descriptors[-1], 
                            "controller":executed_controllers[-1], 
                            "belief_map": belief_map,
                            "belief_action": belief_action,
                            "failed": False})

            if env_real.get_object_position()[2] < 0.5 or step>300:
                trajectory[-1]["failed"] = True
                break

            if np.linalg.norm(getCM() - main_goal) < 0.18:
                print ("Goal reached: ", step)
                if len(all_goals) > 0:
                    main_goal = np.array(all_goals.pop(0))
                    env_real.p.changeVisualShape(goalid, -1, rgbaColor=[1, 0, 1, 0], physicsClientId=env_real.physicsClient)
                    goalid = add_goal(env_real, main_goal)
                else:
                    break
            
            print("Step count: ", step)

    print("Total steps: ", step)
    np.save("trajectory.npy", trajectory)
    np.save("loaded_block_Conditions.npy", blocks)
    np.save("loaded_frictions_Conditions.npy", frictions)
    np.save("loaded_descriptors.npy", descriptors)
    np.save("loaded_parameters.npy", parameters)
    np.save("exp_args.npy", args)
    with open('exp_args.txt', 'w') as file:
        file.write(str(args))
main()
