#Insert packages path to sys.path
#So that the codes can be run from any directory
from pathlib import Path
import os 
import sys
this_file_path = os.path.dirname(os.path.realpath(__file__))
base_path = str(Path(this_file_path).parent)
import pybullet
import pybullet_data
import time 
import numpy as np
import math

class Kuka_pusher_env:
    "Kuka pusher environment"
    def __init__(self, gui=True, toy=-100, objStartPositon = None, objEulerAngles = None, eff_height =1.35, lateral_friction=1, controlStep=0.1, simStep = 0.004, controller=None, jointControlMode = "position", visualizationSpeed = 1.0, boturdf= base_path + "/environments/URDF/kuka_iiwa/kuka_pusher.urdf", floorurdf= base_path + "/environments/URDF/plane.urdf", tableurdf = base_path + "/environments/URDF/table_square/table_square.urdf"):
        assert toy in [0,1,2,3,4,5,6,7,8,9,10,11,12,13], "Only limited choise of obhjects"
        self.p = pybullet
        self.__vspeed = visualizationSpeed 
        self.__init_state= [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 0.0] #position and orientation
        self.__simStep = simStep
        self.__controlStep = controlStep
        self.__controller = controller
        self.physicsClient = object()
        self.eff_height = eff_height
        assert jointControlMode in ["position", "velocity"] 
        self.jointControlMode = jointControlMode #either "position" or "velocity"

        if gui:
            self.physicsClient = self.p.connect(self.p.GUI)
        else:
            self.physicsClient = self.p.connect(self.p.DIRECT)
        
        self.p.setRealTimeSimulation(0, physicsClientId=self.physicsClient)
        self.p.resetSimulation(physicsClientId=self.physicsClient)
        self.p.setTimeStep(self.__simStep, physicsClientId=self.physicsClient)
        self.p.resetDebugVisualizerCamera(4.89, 35.99, -48.6, [0,0,0], physicsClientId=self.physicsClient)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physicsClient)  
        self.p.setGravity(0,0,-10, physicsClientId=self.physicsClient) 
        
        # tableurdf = "./environments/URDF/table_square/table_square.urdf"
        self.__planeId = self.p.loadURDF(floorurdf, [0,0,0], self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        self.__tableId = self.p.loadURDF(tableurdf, [0.5,0,0], self.p.getQuaternionFromEuler([0,0,0]), globalScaling=2.0, physicsClientId=self.physicsClient)
        self.p.changeDynamics(self.__tableId, linkIndex=1, lateralFriction=3.0, physicsClientId=self.physicsClient)
        self.__kukaStartPos = [-1.2, 0, 1.3]
        self.__kukaStartOrientation = self.p.getQuaternionFromEuler([0,0,0]) 
        flags = self.p.URDF_USE_SELF_COLLISION or self.p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        self.__kukaId = self.p.loadURDF(boturdf,self.__kukaStartPos, self.__kukaStartOrientation, useFixedBase=1, flags=flags, globalScaling=2.8, physicsClientId=self.physicsClient) 

        self.objEulerAngles = objEulerAngles if objEulerAngles is not None else [0,0,0]
        self.objStartPositon = None
        self.objStartOrientation = self.p.getQuaternionFromEuler(self.objEulerAngles) 
        self.ObjectId = None
        
        if toy == 0:
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/cube.urdf", self.objStartPositon,  self.p.getQuaternionFromEuler([0,0,0]),  flags=flags, physicsClientId=self.physicsClient)

        if toy == 1:
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/cuboid.urdf", self.objStartPositon,  self.p.getQuaternionFromEuler([0,0,0]),  flags=flags, physicsClientId=self.physicsClient)

        elif toy == 2:
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/cylinder.urdf", self.objStartPositon,  self.p.getQuaternionFromEuler([0,0,0]),  flags=flags, physicsClientId=self.physicsClient)
        elif toy == 3:    
            self.objStartPositon = [0.3, 0.0, 0.98] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/triangle.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 4:   
            self.objStartPositon = [0.3, 0.0, 1.15] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/pentagon.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 5:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/cylinder_ellipse.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 6:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/cylinder_4.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 7:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/cylinder_2.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 8:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/box_pyramid.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 9:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/box_cylinder.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 10:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/bar_rotate_side.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 11:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/bar_rotate_off_centre.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 12:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/lollipop.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        elif toy == 13:   
            self.objStartPositon = [0.3, 0.0, 1.5] if objStartPositon is None else objStartPositon
            self.ObjectId = self.p.loadURDF(base_path + "/environments/URDF/bar_rotate_side_2.urdf", self.objStartPositon, self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)

        self.objectNames = ["cube", "cuboid", "cylinder", "triangle", "pentagon", "cylinder_ellipse", "cylinder_4", "cylinder_2", "box_pyramid", "box_cylinder", "bar_rotate_side", "bar_rotate_side", "lollipop", "bar_rotate_side_2"]
        self.init_position = [-.3, -.4, self.eff_height]
        self.target_orientation = self.p.getQuaternionFromEuler([0, 3.1415, 0]) #rpy:x,y,z 
        self.init_joint_positions = self.p.calculateInverseKinematics (self.__kukaId, 8, self.init_position, self.target_orientation, physicsClientId=self.physicsClient)
        for i in range(7):
            self.p.resetJointState(self.__kukaId, i, self.init_joint_positions[i], physicsClientId=self.physicsClient)        
        
        self.p.changeDynamics(self.__tableId, linkIndex=-1, lateralFriction=lateral_friction, physicsClientId=self.physicsClient)
        self.goal_position = []
        self.goalBodyID = None
        self.visualGoalShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_CYLINDER, radius=0.1, length=0.04, visualFramePosition=[0.,0.,0.], visualFrameOrientation=self.p.getQuaternionFromEuler([0,0,0]) , rgbaColor=[0.0,0.0, 0.0, 0.2], specularColor=[0.5,0.5, 0.5, 1.0], physicsClientId=self.physicsClient)    

    def setFriction(self, lateral_friction):
        self.p.changeDynamics(self.__tableId, linkIndex=-1, lateralFriction=lateral_friction, physicsClientId=self.physicsClient)

    def convert_to_world_coordinate(self, relative_point, frame_pos, frame_angle):
        rm = np.array([[np.cos(np.deg2rad(frame_angle)), -np.sin(np.deg2rad(frame_angle))],
                    [np.sin(np.deg2rad(frame_angle)), np.cos(np.deg2rad(frame_angle))]])
        point = frame_pos + relative_point
        return np.matmul(rm,point)

    def get_object_position(self):
        states = self.p.getBasePositionAndOrientation(self.ObjectId, physicsClientId=self.physicsClient)
        pos = list(states[0])
        # orient = list(states[1])
        return pos
    
    def get_obj_orientation(self):
        states = self.p.getBasePositionAndOrientation(self.ObjectId, physicsClientId=self.physicsClient)
        euler_angles = self.p.getEulerFromQuaternion(list(states[1])) # x,y,z rotation
        return euler_angles

    def set_commands(self, commands, target_euler=None):
        assert commands.size == 3, "x,y,z position of the end effector"
        #Compute the joint positions
        # targetOrientation = self.p.getQuaternionFromEuler([0, 3.1415, 0])
        target_orientation = self.p.getQuaternionFromEuler(target_euler) if target_euler is not None else None
        if target_orientation is not None: 
            target_joint_positions = self.p.calculateInverseKinematics (self.__kukaId, 8, commands, target_orientation, physicsClientId=self.physicsClient)
        else:
            target_joint_positions = self.p.calculateInverseKinematics (self.__kukaId, 8, commands, physicsClientId=self.physicsClient)
        
        if self.jointControlMode == "position":
            # self.p.setJointMotorControlArray(self.__kukaId, [0,1,2,3,4,5,6], self.p.POSITION_CONTROL, target_joint_positions, forces=7*[500], physicsClientId=self.physicsClient)
            for i, pos in enumerate(target_joint_positions):
                self.p.setJointMotorControl2(self.__kukaId, i, self.p.POSITION_CONTROL, pos, maxVelocity=2, force=300, physicsClientId=self.physicsClient)

        elif self.jointControlMode == "velocity":
            max_velocity = 10.0
            current_joint_pos = [s[0] for s in self.p.getJointStates(self.__kukaId, [0,1,2,3,4,5,6], physicsClientId=self.physicsClient)]
            err = np.array(target_joint_positions) - np.array(current_joint_pos)
            target_velocities = np.clip(err * (1.0/ (math.pi * self.__controlStep)), -max_velocity, max_velocity)
            self.p.setJointMotorControlArray(self.__kukaId, [0,1,2,3,4,5,6], self.p.VELOCITY_CONTROL, targetVelocities=target_velocities, physicsClientId=self.physicsClient)

    def run(self, runtime, goal_position = []):
        if (not goal_position == []) and (not self.goal_position == goal_position):             
            if self.goalBodyID is not None:
                self.p.changeVisualShape(self.goalBodyID, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=self.physicsClient)
            self.goalBodyID = self.p.createMultiBody(baseMass=0.0,baseInertialFramePosition=[0,0,0], baseVisualShapeIndex = self.visualGoalShapeId, basePosition = goal_position, useMaximalCoordinates=True, physicsClientId=self.physicsClient)

        self.goal_position = goal_position
        assert not self.__controller == None, "Controller not set"
        # self.__states = [self.getState()] #load with init state
        # self.__rotations = [self.getRotation()]
        self.__commands = []
        old_time = 0
        first = True
        self.__command = object
        controlStep = 0
        self.flip = False
        
        eff_rel_position = self.__controller.getPath()[0]
        frame_pos = self.get_object_position()[0:2]
        frame_angle = self.get_obj_orientation()[2] #z rotation
        eff_position = self.convert_to_world_coordinate(eff_rel_position, frame_pos, frame_angle).tolist() + [self.eff_height]
        self.reset_eff(eff_position)
        
        #debug
        path = self.__controller.getPath()
        p0 = self.convert_to_world_coordinate(path[0], frame_pos, frame_angle).tolist() + [self.eff_height]
        p1 = self.convert_to_world_coordinate(path[-1], frame_pos, frame_angle).tolist() + [self.eff_height]
        self.p.addUserDebugLine(p0, p1, lifeTime=2)
        # cm = self.p.getLinkState(self.ObjectId, 0, physicsClientId=self.physicsClient)
        # print("cm: ", cm[0], " Baes pos: ", self.get_object_position())
        # self.p.addUserDebugLine(list(cm[0])[0:2]+[1.5], list(cm[0])[0:2]+[1.8], lifeTime=5, lineWidth=3, lineColorRGB=[0,0,0])

        for i in range (int(runtime/self.__simStep)): 
            if i*self.__simStep - old_time > self.__controlStep or first:
                # state = self.getState()
                eff_rel_position = self.__controller.nextCommand(i*self.__simStep, runtime)
                eff_position = self.convert_to_world_coordinate(eff_rel_position, frame_pos, frame_angle).tolist() + [self.init_position[2]]
                # print("Eff pos init run: ", eff_position)
                # print("Angle: ", np.rad2deg(self.get_obj_orientation()[2]))
                self.__command = np.array(eff_position)
                # if not first:
                    # self.__states.append(state)
                    # self.__rotations.append(self.getRotation())
                self.__commands.append(self.__command)
                first = False
                old_time = i*self.__simStep
                controlStep += 1
            
            self.set_commands(self.__command)
            self.p.stepSimulation(physicsClientId=self.physicsClient) 
            if self.p.getConnectionInfo(self.physicsClient)['connectionMethod'] == self.p.GUI:
                time.sleep(self.__simStep/float(self.__vspeed)) 
        
    def reset_eff(self, position):
        eff_pos = np.array(self.p.getLinkState(self.__kukaId, 8, physicsClientId=self.physicsClient)[0])
        new_pos = eff_pos.copy()
        new_pos[2] = 2.5
        steps = (np.array(new_pos) - np.array(eff_pos))/10.0
        for i in range(10):
            self.set_commands(np.array(eff_pos)+i*steps)
            self.simulate_for(0.08)

        eff_pos = np.array(self.p.getLinkState(self.__kukaId, 8, physicsClientId=self.physicsClient)[0])
        new_pos = np.array(position).copy()
        new_pos[2] = 2.5
        steps = (np.array(new_pos) - np.array(eff_pos))/10.0 
        for i in range(10):
            self.set_commands(np.array(eff_pos)+i*steps, target_euler=[0, 3.1415, 0])
            self.simulate_for(0.08)

        eff_pos = np.array(self.p.getLinkState(self.__kukaId, 8, physicsClientId=self.physicsClient)[0])
        new_pos = np.array(position).copy()
        steps = (np.array(new_pos) - np.array(eff_pos))/10.0 
        for i in range(10):
            self.set_commands(np.array(eff_pos)+i*steps)
            self.simulate_for(0.08)
    
    def reset(self):
        for i in range(7):
            self.p.resetJointState(self.__kukaId, i, self.init_joint_positions[i], physicsClientId=self.physicsClient)   
        self.p.resetBasePositionAndOrientation(self.ObjectId, self.objStartPositon, self.objStartOrientation, physicsClientId=self.physicsClient)   
        # self.p.resetBasePositionAndOrientation(self.ObjectId, self.objStartPositon, self.objStartOrientation, physicsClientId=self.physicsClient)   

    def simulate_for(self, t):
        for _ in range(int(t/self.__simStep)):
            self.p.stepSimulation(physicsClientId=self.physicsClient) 
            if self.p.getConnectionInfo(self.physicsClient)['connectionMethod'] == self.p.GUI:
                time.sleep(self.__simStep/float(self.__vspeed))

    def setController(self, controller):
        self.__controller=controller

    def getController(self):
        assert self.__controller is not None, "No controller found. Set it first"
        return self.__controller

class GenericController:
    
    "A generic controller. It need to be inherited and overload for specific controllers"
    def __init__(self):
        pass

    def nextCommand(self, CurrenState = np.array([0]), timeStep = 0):
        raise NotImplementedError()
    
    def setParams(self, params = np.array([0])):
        raise NotImplementedError()

    def setRandom(self):
        raise NotImplementedError()

    def getParams(self):   
        return self._params
    
    def setCommandLimits(sef, limits):
        pass

class KukaController (GenericController):
    def __init__(self, params = None, max_final_disp=0.5, scale_intermediate=5.0, array_dim=100, run_time=5):
        # params [0,1] : p values. Length must be even. P1, P2, P3 ....
        # From params a relative bezier will be created
        self.max_disp = max_final_disp
        if params is not None:
            assert len(params)%2 == 0, "Params length must be even"
            self.params = params
        else:
            self.params = None
        self.run_time = run_time
        self.inter_scale = scale_intermediate
        self.array_dim = array_dim
        if params is not None:
            self.bezier_path = self.compute_bezier_path(params, self.array_dim)

    def compute_bezier_path(self, params, array_dim):
        #scale final ponit
        final = params[-2::] * 2.0 * self.max_disp - self.max_disp
        #scale all points
        p = params *2.0 * self.max_disp*self.inter_scale - self.max_disp*self.inter_scale
        # set scaled final point
        p[-1] = final[-1]
        p[-2] = final[-2]
        #add initial point
        p = np.array([0,0] + p.tolist()).reshape(-1, 2)
        n = len(p)-1
        points = []
        step_size = 1.0 / array_dim
        for u in np.arange(0, 1 + step_size, step_size):    
            point = np.zeros(2)
            for i in range(n+1):
                point += math.factorial(n)/(math.factorial(i)*math.factorial(n-i)) * u**i * (1 - u)**(n-i) * p[i]
            point =  np.concatenate((point, np.array([0])), axis=0)
            points.append(point)
        return points[1::]
    
    def nextCommand(self, t):
        index = max(int(len(self.bezier_path) * t / self.run_time) - 1, 0)
        return self.bezier_path[index]
    
    def setParams(self, params):
        self.params = params
        self.bezier_path = self.compute_bezier_path(params, self.array_dim)
    
    def getPath(self):
        return self.bezier_path

class KukaController_line (GenericController):
    def __init__(self, params = None, n_points=100, object_radius=0.5):
        self.n = n_points
        self.r = object_radius
        
        if params is not None:
            assert len(params) == 2, "Params length must be 2"
            self.params = params
        else:
            self.params = None
        if params is not None:
            self.line_path = self.compute_line_path(params)

    def compute_line_path(self, params):
        #returnd path in objects coordinate frame
        #Need to convet to world coordinate before applying to the robot
        theta0 = np.deg2rad((params[0]*2.0 - 1.0) * 180.0)
        theta1 = np.deg2rad((params[1]*2.0 - 1.0) * 180.0)
        p0 = np.array([self.r*np.cos(theta0), self.r*np.sin(theta0)])
        p1 = np.array([self.r*np.cos(theta1), self.r*np.sin(theta1)])
        # diff = np.linalg.norm(p0-p1)
        # dist = diff/self.n
        points = [p0]
        step = (p1-p0)/self.n
        for _ in range(self.n):
            points.append(points[-1]+step)
        return points
    
    def nextCommand(self, t, run_time):
        index = min(max(int(len(self.line_path) * t / run_time) - 1, 0), len(self.line_path)-1)
        # print ("Index: ", index)
        return self.line_path[index]
    
    def setParams(self, params):
        self.params = params
        self.line_path = self.compute_line_path(params)
    
    def getPath(self):
        return self.line_path

if __name__ == '__main__':
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
    
    files = ["/home/rkaushik/projects/cluster_sync/MAP_Elites_exps/kuka_pusher_prior/kuka_pusher_map_toy_1_friction_1/archive_map0_3000.dat"]
    params, fitness, desc = load_map(files)

    controller = KukaController_line(params = np.array([0.3, 0.8]), object_radius=0.5) 
    env = Kuka_pusher_env(gui=True, toy=6, controller=controller, jointControlMode = "position", visualizationSpeed = 1.0, lateral_friction=0., objStartPositon=[0.3, -1.0, 1.5])
    parameters = []
    positions = []
    for i in range(1000):
        pp = params[np.random.randint(0, len(params))]
        parameters.append(pp)
        controller.setParams(pp)
        xy_old = np.array(env.get_object_position())[0:2]
        env.run(2)
        xy_new = np.array(env.get_object_position())[0:2]
        print ("old: ", xy_old, " new: ", xy_new)
        print("Displacement: ", xy_new - xy_old) 
        positions.append(xy_new - xy_old)
        env.reset()
    plt.plot([d[0] for d in positions], [d[1] for d in positions], 'ob')
    plt.show()
    np.save("positions.npy", positions)
    np.save("parameters.npy", parameters)
    time.sleep(1000)