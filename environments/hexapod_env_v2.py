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

class Hexapod_env:
    "Hexapod environment"
    def __init__(self, gui, controlStep=0.1, simStep = 0.015, controller=None, jointControlMode = "position", visualizationSpeed = 1.0, boturdf= base_path + "/environments/URDF/pexod.urdf", floorurdf= base_path + "/environments/URDF/plane.urdf", lateral_friction=10.0):
        self.p = pybullet
        self.__vspeed = visualizationSpeed 
        self.__init_state= [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 0.0] #position and orientation
        self.__simStep = simStep
        self.__controlStep = controlStep
        self.__controller = controller
        self.physicsClient = object()
        assert jointControlMode in ["position", "velocity"] 
        self.jointControlMode = jointControlMode #either "position" or "velocity"
        self.__base_collision = False
        self.__heightExceed = False

        if gui:
            self.physicsClient = self.p.connect(self.p.GUI)
        else:
            self.physicsClient = self.p.connect(self.p.DIRECT)
        
        self.p.resetSimulation(physicsClientId=self.physicsClient)
        self.p.setTimeStep(self.__simStep, physicsClientId=self.physicsClient)
        self.p.resetDebugVisualizerCamera(1.5, 50, -35.0, [0,0,0], physicsClientId=self.physicsClient)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physicsClient)  
        self.p.setGravity(0,0,-10.0, physicsClientId=self.physicsClient) 
        
        self.__planeId = self.p.loadURDF(floorurdf, [0,0,0], self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)
        self.hexapodStartPos = [0,0,0.2] # Start at collision free state. Otherwise results becomes a bit random
        self.hexapodStartOrientation = self.p.getQuaternionFromEuler([0,0,0]) 
        flags= self.p.URDF_USE_INERTIA_FROM_FILE or self.p.URDF_USE_SELF_COLLISION or self.p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        self.__hexapodId = self.p.loadURDF(boturdf,self.hexapodStartPos, self.hexapodStartOrientation, useFixedBase=0, flags=flags, physicsClientId=self.physicsClient) 

        self.joint_list = self._make_joint_list(self.__hexapodId)
        self.goal_position = []
        self.goalBodyID = None
        self.visualGoalShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_CYLINDER, radius=0.2, length=0.04, visualFramePosition=[0.,0.,0.], visualFrameOrientation=self.p.getQuaternionFromEuler([0,0,0]) , rgbaColor=[0.0,0.0, 0.0, 0.5], specularColor=[0.5,0.5, 0.5, 1.0], physicsClientId=self.physicsClient)    

        self.p.changeDynamics(self.__planeId, linkIndex=-1, lateralFriction=lateral_friction, physicsClientId=self.physicsClient)

    def setFriction(self, lateral_friction):
        self.p.changeDynamics(self.__planeId, linkIndex=-1, lateralFriction=lateral_friction, physicsClientId=self.physicsClient)
        
    def get_simulator(self):
        return self.p, self.physicsClient

    def _make_joint_list(self, botId):
        joint_names = [b'body_leg_0', b'leg_0_1_2', b'leg_0_2_3',
        b'body_leg_1', b'leg_1_1_2', b'leg_1_2_3',
        b'body_leg_2', b'leg_2_1_2', b'leg_2_2_3',
        b'body_leg_3', b'leg_3_1_2', b'leg_3_2_3',
        b'body_leg_4', b'leg_4_1_2', b'leg_4_2_3',
        b'body_leg_5', b'leg_5_1_2', b'leg_5_2_3',
        ]
        joint_list = []
        for n in joint_names:
            for joint in range (self.p.getNumJoints(botId, physicsClientId=self.physicsClient)):
                name = self.p.getJointInfo(botId, joint, physicsClientId=self.physicsClient)[1]
                if name == n:
                    joint_list += [joint]
        return joint_list
    
    def run(self, runtime, goal_position = []):
        
        if (not goal_position == []) and (not self.goal_position == goal_position):             
            # if self.goalBodyID is not None:
            #     self.p.removeBody(self.goalBodyID, physicsClientId=self.physicsClient)
            self.goalBodyID = self.p.createMultiBody(baseMass=0.0,baseInertialFramePosition=[0,0,0], baseVisualShapeIndex = self.visualGoalShapeId, basePosition = goal_position, useMaximalCoordinates=True, physicsClientId=self.physicsClient)
            # print ("Goal information: ", self.goalBodyID)
        self.__base_collision = False
        self.__heightExceed = False
        self.goal_position = goal_position
        assert not self.__controller == None, "Controller not set"
        self.__states = [self.getState()] #load with init state
        self.__rotations = [self.getRotation()]
        self.__commands = []
        old_time = 0
        first = True
        self.__command = object
        controlStep = 0
        self.flip = False
        for i in range (int(runtime/self.__simStep)): 
            if i*self.__simStep - old_time > self.__controlStep or first:
                state = self.getState()
                self.__command = self.__controller.nextCommand(i*self.__simStep)
                if not first:
                    self.__states.append(state)
                    self.__rotations.append(self.getRotation())
                self.__commands.append(self.__command)
                first = False
                old_time = i*self.__simStep
                controlStep += 1
            
            self.set_commands(self.__command)
            self.p.stepSimulation(physicsClientId=self.physicsClient) 
            if self.p.getConnectionInfo(self.physicsClient)['connectionMethod'] == self.p.GUI:
                time.sleep(self.__simStep/float(self.__vspeed)) 
            
            # Flipfing behavior
            if self.getZOrientation() < 0.0:
                self.flip=True
            
            #Base floor collision
            if len(self.p.getContactPoints(self.__planeId, self.__hexapodId, -1, -1, physicsClientId=self.physicsClient)) > 0:
                self.__base_collision=True
            
            #Jumping behavior when CM crosses 2.2
            if self.getState()[2] > 2.2: 
                self.__heightExceed = True

    def isBaseCollision(self):
        return self.__base_collision         
    
    def isHeightExceed(self):
        return self.__heightExceed

    def states(self):
        return self.__states 
    
    def rotations(self):
        return self.__rotations 

    def commands(self):
        return self.__commands

    def simStep(self):
        return self.__simStep
    
    def controlStep(self):
        return self.__controlStep

     #command should be numpy array
    def set_commands(self, commands):
        assert commands.size == len(self.joint_list), "Command length doesn't match with controllable joints"
        counter = 0
        for joint in self.joint_list:
            info = self.p.getJointInfo(self.__hexapodId, joint, physicsClientId=self.physicsClient)
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]
            pos = min(max(lower_limit, commands[counter]), upper_limit)
            
            if self.jointControlMode == "position":
                self.p.setJointMotorControl2(bodyUniqueId=self.__hexapodId, jointIndex=joint, 
                controlMode=self.p.POSITION_CONTROL, 
                targetPosition = commands[counter], 
                force=max_force, 
                maxVelocity=max_velocity, 
                physicsClientId=self.physicsClient)

            elif self.jointControlMode == "velocity":
                current_joint_pos = self.p.getJointState(bodyUniqueId=self.__hexapodId, jointIndex=joint,physicsClientId=self.physicsClient)[0]
                err = pos - current_joint_pos
                self.p.setJointMotorControl2(bodyUniqueId=self.__hexapodId, jointIndex=joint, 
                    controlMode=self.p.VELOCITY_CONTROL, 
                    # velocity must be limited as it it not done automatically
                    # max_force is however considered here
                    targetVelocity = np.clip(err * (1.0/ (math.pi * self.__controlStep)), -max_velocity, max_velocity),
                    force=max_force, 
                    physicsClientId=self.physicsClient)
            counter = counter + 1

    def getState(self):
        '''
        Returns the position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order.
        Use self.p.getEulerFromQuaternion to convert the quaternion to Euler if needed.
        '''
        states = self.p.getBasePositionAndOrientation(self.__hexapodId, physicsClientId=self.physicsClient)
        pos = list(states[0])
        orient = list(states[1])
        return pos + orient
    
    def getZOrientation(self):
        '''
        Returns z component of up vector of the robot
        It is negative if the robot is flipped.
        '''
        states = self.p.getBasePositionAndOrientation(self.__hexapodId, physicsClientId=self.physicsClient)
        z_componentOfUp = self.p.getMatrixFromQuaternion(list(states[1]))[-1]
        return z_componentOfUp
    
    def getRotation(self):
        '''
        Returns rotation matrix
        '''
        states = self.p.getBasePositionAndOrientation(self.__hexapodId, physicsClientId=self.physicsClient)
        return self.p.getMatrixFromQuaternion(list(states[1]))
    
    def getEulerAngles(self):
        states = self.p.getBasePositionAndOrientation(self.__hexapodId, physicsClientId=self.physicsClient)
        euler_angles = self.p.getEulerFromQuaternion(list(states[1])) # x,y,z rotation
        return np.rad2deg(euler_angles)
    
    def flipped(self):
        return self.flip
        
    def reset(self, startPosition=None, startOrientation=None):
        '''
        orinetation must be in quarternion
        ''' 
        for i in range(self.p.getNumJoints(self.__hexapodId, physicsClientId=self.physicsClient)):
            self.p.resetJointState(self.__hexapodId, i, 0.0, 0.0, physicsClientId=self.physicsClient)
        
        pos = startPosition if startPosition is not None else self.hexapodStartPos
        orient = startOrientation if startOrientation is not None else self.hexapodStartOrientation
        self.p.resetBasePositionAndOrientation(self.__hexapodId, pos, orient, physicsClientId=self.physicsClient)
        
        self.__states = [] 
        self.__rotations = []
        self.__commands = [] 

    def disconnet(self):
        self.p.disconnect(physicsClientId=self.physicsClient)

    def setController(self, controller):
        self.__controller = controller
    
    def getController(self):
        return self.__controller
    # def commandDim(self):
    #     return 7


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

class HexaController (GenericController) :
    def __init__(self, params=None, array_dim=100):
        self.array_dim = array_dim
        self._params = None

    def nextCommand(self, t):
        freq = 1.0 #Hz
        k = int(math.floor(t * self.array_dim * freq)) % self.array_dim  #1/freq second  => full array_dim, therefore 1 second => (array_dim*freq) % array_dim
        return self.trajs[:, k]
    
    def _compute_trajs(self, p, array_dim):
        trajs = np.zeros((6 * 3, array_dim))
        k = 0
        for i in range(0, 36, 6):
            trajs[k,:] =  0.5 * self._control_signal(p[i], p[i + 1], p[i + 2], array_dim)
            trajs[k+1,:] = self._control_signal(p[i + 3], p[i + 4], p[i + 5], array_dim)
            trajs[k+2,:] = trajs[k+1,:] 
            k += 3
        return trajs * math.pi / 4.0
        
    def _control_signal(self, amplitude, phase, duty_cycle, array_dim=100):
        '''
        create a smooth periodic function with amplitude, phase, and duty cycle, 
        amplitude, phase and duty cycle are in [0, 1]
        '''
        assert(amplitude >= 0 and amplitude <= 1)
        assert(phase >= 0 and phase <= 1)
        assert(duty_cycle >= 0 and duty_cycle <= 1)
        command = np.zeros(array_dim)

        # create a 'top-hat function'
        up_time = array_dim * duty_cycle
        temp = [amplitude if i < up_time else -amplitude for i in range(0, array_dim)]
        
        # smoothing kernel
        kernel_size = int(array_dim / 10)
        kernel = np.zeros(int(2 * kernel_size + 1))
        sigma = kernel_size / 3
        for i in range(0, len(kernel)):
            kernel[i] =  math.exp(-(i - kernel_size) * (i - kernel_size) / (2 * sigma**2)) / (sigma * math.sqrt(math.pi))
        sum = np.sum(kernel)

        # smooth the function
        for i in range(0, array_dim):
            command[i] = 0
            for d in range(1, kernel_size + 1):
                if i - d < 0:
                    command[i] += temp[array_dim + i - d] * kernel[kernel_size - d]
                else:
                    command[i] += temp[i - d] * kernel[kernel_size - d]
            command[i] += temp[i] * kernel[kernel_size]
            for d in range(1, kernel_size + 1):
                if i + d >= array_dim:
                    command[i] += temp[i + d - array_dim] * kernel[kernel_size + d]
                else:
                    command[i] += temp[i + d] * kernel[kernel_size + d]
            command[i] /= sum

        # shift according to the phase
        final_command = np.zeros(array_dim)
        start = int(math.floor(array_dim * phase)) #in python 2, floor returns float type
        current = 0
        for i in range(start, array_dim):
            final_command[current] = command[i]
            current += 1
        for i in range(0, start):
            final_command[current] = command[i]
            current += 1
            
        assert(len(final_command) == array_dim)
        return final_command

    def setParams(self, params, array_dim=100):
        self._params = params
        self.array_dim = array_dim
        self.trajs = self._compute_trajs(params, array_dim)

    def setRandom(self):
        self._random = True
        self.setParams(np.random.rand(36))

    def getParams(self):  
        return self._params
    
    def setCommandLimits(self, limits):
        pass
    
    def getJointTrajectory(self):
        assert self._params is not None, "No control parameters to compute joint trajectory"
        return self.trajs
    
    def setJointTrajectory(self, traj):
        self.trajs = traj

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    def block(pp,blocks):
        for i in blocks:
            pp[i*6] = 0 
            pp[i*6 + 3] = 0 
        return pp

    env = Hexapod_env(gui=True, visualizationSpeed=1.0, simStep=0.004, controlStep=0.01, jointControlMode = "velocity", lateral_friction=5)
    ctlr = HexaController()
    sim_time = 3.0
    env.setController(ctlr)
    x = []
    pp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.65, 0.3, 0.95, 0.95, 0.4, 0.9, 0.3, 0.4, 0.65, 0.45, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.55, 0.65, 0.95, 0.95, 0.2, 0.45, 0.95, 0.45, 0.9, 0.45, 0.5]
    # Current obs:  [-0.44416973 -0.0514075 ]
    # expected obs:  [-0.47388806 -0.19579347]
    pp = block(pp, [0,3])
    ctlr.setParams(pp)
    for i in range(1):
        ctlr.setParams(36*[0])
        cm = env.getState()[0:2]
        env.run(1.0)
        
        ctlr.setParams(pp)
        env.run(sim_time)
        
        states = env.states()
        rew = states[-1][0]
        final_state = states[-1][0:2]
        print(np.array(final_state)-np.array(cm))
        print(final_state)
        print("")
        env.reset()