#Insert packages path to sys.path
#So that the codes can be run from any directory
from pathlib import Path
import os 
import sys
this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(this_file_path).parent))
import numpy as np
import math
from environments.hexapod_env_v2 import * 
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
        'blocked_legs': [],
        'gui': False,
        'visualization_speed': 2.0,
        'search_size': 300, #Pick top search_size number of controllers from the archives
        'kernel_l': 1.0,
        'kernel_var': 0.3, #Angle range to search for high performing behavior  
        'lateral_friction': 1.0,
        'archives_path': "./data/hexapod_prior",
        'mapelites_generations': 5000,
        'ablation': None} # "only_ucb", "only_learning"

#optional arguments
parser = argparse.ArgumentParser()

parser.add_argument("--archives_path", 
                    help='Path to MAP-elites archives', 
                    type=str)

parser.add_argument("--mapelites_generations", 
                    help='MAP-elites generations used. Required to decide the file names to load from', 
                    type=int)

parser.add_argument("--ablation", 
                    help=' "only_ucb" or "only_learning"', 
                    choices=['only_ucb','only_learning'],
                    type=str)

parser.add_argument("--directions", 
                    help='Desired mini archive directions', 
                    type=int)

parser.add_argument("--ucb_const", 
                    help='ucb constant multiplier parameter', 
                    type=float)

parser.add_argument("--blocked_legs", 
                    help="Inddices of the legs to be blocked", 
                    type= int,
                    nargs='+')

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
if arguments.blocked_legs is not None: args['blocked_legs'] = arguments.blocked_legs
if arguments.gui is not None: args['gui'] = arguments.gui
if arguments.visualization_speed is not None: args['visualization_speed'] = arguments.visualization_speed
if arguments.kernel_l is not None: args['kernel_l'] = arguments.kernel_l
if arguments.kernel_var is not None: args['kernel_var'] = arguments.kernel_var
if arguments.search_size is not None: args['search_size'] = arguments.search_size
if arguments.lateral_friction is not None: args['lateral_friction'] = arguments.lateral_friction
if arguments.ablation is not None: args['ablation'] = arguments.ablation

sim_time = 3.0
env_real = Hexapod_env(gui=args['gui'], visualizationSpeed=args['visualization_speed'], simStep=0.004, controlStep=0.01, jointControlMode="velocity", lateral_friction=args["lateral_friction"])
env_real.p.resetDebugVisualizerCamera(6.0, 90, -70.0, [-2,4,0], physicsClientId=env_real.physicsClient)
ctlr = HexaController()
env_real.setController(ctlr)


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

def disable_leg_param(param, blocked_legs):
    pp = param.copy()
    for leg in blocked_legs:
        assert leg < 6 and leg >= 0
        # disable first joint
        pp[6*leg] = 0   
        pp[6*leg+1] = 0   
        pp[6*leg+2] = 0  
        
        # disable 2nd joint
        pp[6*leg+3] = 0   
        pp[6*leg+4] = 0   
        pp[6*leg+5] = 0   
    return pp

def execute_real_bot(param, runtime=10, visual_goal=[], blocked_legs=[], lateral_friction=None):
    pp = param.copy()
    for leg in blocked_legs:
        assert leg < 6 and leg >= 0
        # disable first joint
        pp[6*leg] = 0   
        pp[6*leg+1] = 0   
        pp[6*leg+2] = 0  
        
        # disable 2nd joint
        pp[6*leg+3] = 0   
        pp[6*leg+4] = 0   
        pp[6*leg+5] = 0   
    
    if lateral_friction is not None:
        env_real.setFriction(lateral_friction)    
    env_real.getController().setParams(pp)
    env_real.run(runtime, goal_position=visual_goal)

    final_state = env_real.getState()[0:2]    
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

def get_direction(robot_cm, robot_direction,  goal):
    cm2goal = goal - robot_cm
    cm2goal_polar = convert_cart2polar(cm2goal)

    theta = np.deg2rad(cm2goal_polar[1] - robot_direction)
    theta_reduced = np.arctan2(np.sin(theta),np.cos(theta))

    return np.rad2deg(theta_reduced)

def getFaceAngle():
    final_face_vec = [env_real.getRotation()[0], env_real.getRotation()[3]]
    theta = convert_cart2polar(final_face_vec)[1]
    return theta 

def getCM():
    return np.array(env_real.getState()[0:2])
    
def relative_coordinate(absolute_point, face_angle, cm):
    #World frame to robots frame
    rm = np.array([[np.cos(np.deg2rad(face_angle)), np.sin(np.deg2rad(face_angle))],
                    [-np.sin(np.deg2rad(face_angle)), np.cos(np.deg2rad(face_angle))]])
    point = absolute_point - cm
    return np.matmul(rm,point)

def add_obstacles(obstacles):
    for i, obstacle in enumerate(obstacles):
        colBoxId = env_real.p.createCollisionShape(env_real.p.GEOM_BOX,halfExtents=[(obstacle[1]-obstacle[0])/2.0, (obstacle[3]-obstacle[2])/2.0, 0.2],physicsClientId=env_real.physicsClient)
        env_real.p.createMultiBody(baseMass=30, baseCollisionShapeIndex = colBoxId, basePosition = [(obstacle[0]+obstacle[1])/2.0, (obstacle[2]+obstacle[3])/2.0, 0.2],physicsClientId=env_real.physicsClient)
        if i > len(obstacles)-5: #Box obstacles
            visBoxId = env_real.p.createVisualShape(env_real.p.GEOM_BOX,halfExtents=[(obstacle[1]-obstacle[0])/2.0+0.005, (obstacle[3]-obstacle[2])/2.0+0.005, 0.2+0.005],rgbaColor=[0.1,0.7,0.6, 1.0], specularColor=[0.3, 0.5, 0.5, 1.0],physicsClientId=env_real.physicsClient)
            env_real.p.createMultiBody(baseMass=0, baseInertialFramePosition=[0,0,0], baseVisualShapeIndex = visBoxId, useMaximalCoordinates=True, basePosition = [(obstacle[0]+obstacle[1])/2.0, (obstacle[2]+obstacle[3])/2.0, 0.2],physicsClientId=env_real.physicsClient)

def add_goal(goal):
    visualGoalShapeId = env_real.p.createVisualShape(shapeType=env_real.p.GEOM_CYLINDER, radius=0.5, length=0.04, visualFramePosition=[0.,0.,0.], visualFrameOrientation=env_real.p.getQuaternionFromEuler([0,0,0]) , rgbaColor=[0.0,0.9, 0.8, 0.8], specularColor=[0.5,0.9, 0.9, 1.0], physicsClientId=env_real.physicsClient)    
    env_real.p.createMultiBody(baseMass=0.0,baseInertialFramePosition=[0,0,0], baseVisualShapeIndex = visualGoalShapeId, basePosition = [goal[0], goal[1], 0.03], useMaximalCoordinates=True, physicsClientId=env_real.physicsClient)


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
    # ticks = tuple([str(d) for d in leg_blocks])
    # plot_map_ucb(prior_map_means, ucb, ticks)
    return posterior 

def gaussian_likelihood(means, vars, point):
    #TODO: Need to test
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
    for dd in maps_dirs:
        with open(dd+"/env_params.txt", 'r') as f:
            s = f.readlines()[0]
        json_acceptable_string = s.replace("'", "\"")
        d = json.loads(json_acceptable_string)
        if args["ablation"] == "only_learning":
            if not (args["blocked_legs"] == d["blocked_legs"]) and d["blocked_legs"] in [[]] and d["lateral_friction"]==5: #NOTE: load undamaged prior for friction 5 
                blocks.append(d["blocked_legs"])
                frictions.append(d["lateral_friction"])
                map_files.append(dd+"/archive_map0_5000.dat")
        else:
            if not (args["blocked_legs"] == d["blocked_legs"]): #NOTE: Else load all maps  
                blocks.append(d["blocked_legs"])
                frictions.append(d["lateral_friction"])
                map_files.append(dd+"/archive_map0_5000.dat")

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

    for damage_index, params in enumerate(parameters):
        for i, pp in enumerate(params):
            parameters[damage_index][i] = disable_leg_param(pp, blocks[damage_index])

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
    Obstacles = np.array([[-6, -1, -1, 1.0], [-3, 1, 3, 4], \
                            [1, 1.5, -1.5, 8], 
                            [-6.5, 1, -1.5, -1.0], 
                            [-6.5, -6.0, -1, 8],
                            [-6.5, 1.5, 8, 8.5]]) #[[xmin, xmax, ymin, ymax], ...]
    
    Obstacles_offset = Obstacles + np.array([-0.5, 0.5, -0.5, 0.5])
    
    boundary = [-6, 6, -8, 8]
    # boundary = [-20, 20, -20, 20]
    # main_goal= np.array([-4, 2.3])
    main_goal= np.array([-4+2, 2.3+3])
    add_goal(main_goal)
    add_obstacles(Obstacles)
    step = 0

    prior_log_probabilities = np.log(len(map_files)*[1/float(len(map_files))])
    last_log_likelihoods = np.log(np.ones(len(map_files)))

    executed_controllers = []
    executed_descriptors = []
    observations = []

    executed_descriptors_mapwise = [[] for i in range(len(map_files))]
    observations_mapwise = [[] for i in range(len(map_files))]

    prior_map_index = []
    prior_map_trials = np.ones(len(map_files))
    prior_map_means = np.ones(len(map_files))*0.6

    for test in range(1):
        env_real.reset()
        pp = np.random.rand(36)*0.0
        obs = execute_real_bot(pp, 1)
        bot_directon = env_real.getEulerAngles()[2]
        bot_cm = getCM()
        first = True
        step = 0

        trajectory = []
        belief_map = []
        belief_action = []
        trajectory.append({"bot_cm":bot_cm, "bot_directon":bot_directon, "step":step, "action": None, "controller": None, "belief_map": None, "belief_action": None})
        
        planner = A_star(Obstacles_offset, boundary, resolution=0.05)
        main_path = planner.plan(bot_cm, main_goal)
        sub_goals = main_path[0:len(main_path):20]
        sub_goals.append(main_goal.tolist())
        _ = sub_goals.pop(0) #Remove the initi position
        sub_goal = sub_goals.pop(0)

        sub_goal_reached = False
        while True: 
            if sub_goal_reached ==True:
                sub_goal_reached =False
                sub_goal = sub_goals.pop(0) if len(sub_goals) > 0 else main_goal

            step +=1
            _ = execute_real_bot(36*[0], 1.0)
            bot_full_orientation = env_real.getEulerAngles()
            bot_directon = env_real.getEulerAngles()[2]
            bot_cm = getCM()
            
            #plan path
            planner = A_star(Obstacles, boundary, resolution=0.05)
            path = planner.plan(bot_cm, sub_goal)
            goal = np.array(path[17]) if len(path) > 17 else sub_goal
            visual_goal = [goal[0], goal[1], 0.01]        
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
                        # print("Mean: ", gp_models[prior_index].predict(np.array([test_controllers[-1]]))[0])
                    pp = test_controllers[np.argmax(likelihoods)]
                    dd = test_descriptors[np.argmax(likelihoods)]
                    prior_map_trials[np.argmax(likelihoods)] += 1.0
                    prior_map_index.append(np.argmax(likelihoods))
                    belief_action = {"block_belief": blocks[np.argmax(likelihoods)] , "friction_belief": frictions[np.argmax(likelihoods)], "sorted_actions": test_descriptors, "full_action_posterior": likelihoods}
                    executed_descriptors.append(dd)
                    executed_descriptors_mapwise[prior_map_index[-1]].append(dd)

            #Execute bot
            # obs = execute_real_bot(pp, 3, visual_goal=visual_goal, blocked_legs=args['blocked_legs'])
            obs = execute_real_bot(pp, 3, blocked_legs=args['blocked_legs'])
            _ = execute_real_bot(36*[0], 1.0)
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
                            "bot_cm":bot_cm, 
                            "bot_directon":bot_directon, 
                            "step":step, 
                            "action": executed_descriptors[-1], 
                            "controller":executed_controllers[-1], 
                            "belief_map": belief_map,
                            "belief_action": belief_action})

            if env_real.flipped():
                print("Flipped !! Resetting at that point")
                simulator, _ = env_real.get_simulator() 
                start_pos = env_real.hexapodStartPos
                start_pos[0] = obs[0]
                start_pos[1] = obs[1]
                start_orient = simulator.getQuaternionFromEuler(np.deg2rad(bot_full_orientation))
                env_real.reset(start_pos, start_orient)

            if np.linalg.norm(getCM() - main_goal) < 0.3:
                print ("Goal reached: ", step)
                break
            elif np.linalg.norm(getCM() - np.array(sub_goal)) < 1.0:
                sub_goal_reached = True
                print("Sub goal reached")

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
