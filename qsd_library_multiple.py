import numpy as np
import tensorflow as tf
import scipy as sp

def cum_kron(rp):
    r = np.array([[1]])
    for i in range(len(rp)):
        r = np.kron(r, rp[i])
    return r

def prob_outcome(rho, POVM, j):
    pvec = np.zeros((len(rho), len(POVM)))
    for k in range(len(rho)):
        for i in range(len(POVM)):
            pvec[k][i] = np.abs(np.trace(rho[k][j]@POVM[i]))
    return pvec
     
def outcome(state, pvec_matrix):
    dist= pvec_matrix[state]
    out = np.random.choice(dist.shape[0], p= dist.tolist())
    return out


def PMstate(rho, POVM, q, j, outcome):
    denom = 0
    for k in range(len(rho)):
        denom += q[k]*np.trace(POVM[outcome]@rho[k][j])
    for k in range(len(rho)):
        q[k] = q[k]*np.trace(POVM[outcome]@rho[k][j])/denom + 10e-13
    return q


def isdone(obs_state, d):
    x = True
    for _ in range(len(d)):
        if int(obs_state[_])==0:
            x = False
    return x
    
def generate_initial_state(d, m, depolarized = True):
    rho = np.tile(np.tile(np.eye(2), (len(d), 1, 1)), (m,1,1,1))
    for k in range(m):
        t = np.array([np.random.random(1)[0] for _ in range(m)])
        for j in range(len(d)):
            rho[k][j] = np.array([[t[k]**2, t[k]*np.sqrt(1-t[k]**2)],[t[k]*np.sqrt(1-t[k]**2), 1-t[k]**2]])
            if depolarized == True:
                g = random.random()
                rho[k][j] = rho[k][j]*(1-g)+g/2*np.eye(2)
    p = []
    for k in range(m-1):
        p.append(np.random.uniform(0, 1-sum(p), 1)[0]) 
    p.append(1-sum(p))
    q = np.random.permutation(np.array(p))
    return rho, q

import numpy as np
import gym
import scipy as sp
from itertools import chain

import cmath
import copy
import random 

def POVMsimple(x):
    return np.array([[[x**2, x*np.sqrt(1-x**2)],[x*np.sqrt(1-x**2), 1-x**2]],
                     [[1-x**2, -x*np.sqrt(1-x**2)],[-x*np.sqrt(1-x**2), x**2]]])


def POVMU(x):
    return np.array([[np.cos(np.pi*x), -np.sin(np.pi*x)],[np.sin(np.pi*x), np.cos(np.pi*x)]])

def POVMtU(x):
    return np.array([[np.cos(np.pi*x), np.sin(np.pi*x)],[-np.sin(np.pi*x), np.cos(np.pi*x)]])

def POVMtsimple(x):
    U = np.array([[np.cos(np.pi*2/(3)), -np.sin(np.pi*2/(3))],[np.sin(np.pi*2/(3)), np.cos(np.pi*2/(3))]])
    Ut = np.array([[np.cos(np.pi*2/(3)), np.sin(np.pi*2/(3))],[-np.sin(np.pi*2/(3)), np.cos(np.pi*2/(3))]])
    return 2/3*np.array([POVMU(x)@U@np.array([[1,0],[0,0]])@Ut@POVMtU(x), POVMU(x)@U@U@np.array([[1,0],[0,0]])@Ut@Ut@POVMtU(x),  POVMU(x)@np.array([[1,0],[0,0]])@POVMtU(x)])

def act_space_map(action_label, quant):
    j = int(action_label//quant)
    actlabel= int(action_label)-quant*j
    return np.array([actlabel, j])


import gym
import copy


class QSDEnv(gym.Env):
    metadata = {'render.modes': []}
    reward_range = (float('-0.5'), float('1'))
    spec = None
    
    def __init__(self, env_config):
        self.d = env_config["d"]   
        self.rho = copy.copy(env_config["rho"])
        self.rho_init = copy.copy(self.rho)
        self.q= env_config["q"]
        self.q_init = copy.copy(self.q)
        self.separable = env_config["separable"]
        self.quantization = env_config["quantization"]
        self.m = len(self.rho)
        
        vec = []
        for x in range(self.m):
            vec.append(x)
        self.state = np.random.choice(vec, 1, p = self.q_init.tolist())[0]
        self.obs_state = np.zeros(len(self.d), dtype = np.int8)
        self.done = False
        
        
        vec = [gym.spaces.Box(low = 0, high = 1, shape = (), dtype = np.float32) for _ in range(self.m-1)]
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(0, 1, shape=(len(self.d), )), gym.spaces.Box(0, 1, shape=(self.m-1, ))
                                                  ))
        self.action_space = gym.spaces.Discrete(len(self.d)*self.quantization)
        
        
    def get_obs(self):
         return tuple([self.obs_state, self.q[:self.m-1]])

        
    def reset(self):
        self.q = copy.copy(self.q_init)
        vec = []
        for x in range(self.m):
            vec.append(x)
            
        self.state = np.random.choice(vec, 1, p = self.q_init.tolist())[0]
        self.rho= copy.copy(self.rho_init)
        self.obs_state = np.zeros(len(self.d))
        self.done = False
        return self.get_obs()
        
    def render(self):
        pass
        
    def get_reward(self, y):
        if self.done==True and np.argmax(self.q)==self.state:
            return 1
        elif y == 1:
            return -0.3
        else:
            return 0
       
    def step(self, action_label):
        act = act_space_map(action_label, self.quantization)
        POVM, j = POVMsimple(act[0]/self.quantization), act[1]
        y=0
        if self.obs_state[j]==1:
            y=1
        self.take_action(POVM, j)
        self.obs_state[j] = int(1)
        self.done = isdone(self.obs_state, self.d)
        self.reward = self.get_reward(y)
        return self.get_obs(), self.reward, self.done, {}
        
        
    def take_action(self, POVM, j):
        pvec_matrix = prob_outcome(self.rho, POVM, j)
        out = outcome(self.state, pvec_matrix)
        self.q = PMstate(self.rho, POVM, self.q, j, out)
        self.done = isdone(self.obs_state, self.d)
        
class TrineEnv(gym.Env):
    metadata = {'render.modes': []}
    reward_range = (float('-0.5'), float('1'))
    spec = None
    
    def __init__(self, env_config):
        self.d = env_config["d"]   
        self.rho = copy.copy(env_config["rho"])
        self.rho_init = copy.copy(self.rho)
        self.q= env_config["q"]
        self.q_init = copy.copy(self.q)
        self.separable = env_config["separable"]
        self.quantization = env_config["quantization"]
        self.m = len(self.rho)
        
        vec = []
        for x in range(self.m):
            vec.append(x)
        self.state = np.random.choice(vec, 1, self.q_init.tolist())[0]
        self.obs_state = np.zeros(len(self.d), dtype = np.int8)
        self.done = False
        
        
        vec = [gym.spaces.Box(low = 0, high = 1, shape = (), dtype = np.float32) for _ in range(self.m-1)]
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(0, 1, shape=(len(self.d), )), gym.spaces.Box(-0.0000001, 1.0000001, shape=(self.m-1, ))
                                                  ))
        self.action_space = gym.spaces.Discrete(len(self.d)*self.quantization)
        
        
    def get_obs(self):
         return tuple([self.obs_state, self.q[:self.m-1]])
        
    def reset(self):
        self.q = copy.copy(self.q_init) 
        vec = []
        for x in range(self.m):
            vec.append(x)
            
        self.state = np.random.choice(vec, 1, self.q_init.tolist())[0]
        self.rho= copy.copy(self.rho_init)
        self.obs_state = np.zeros(len(self.d))
        self.done = False
        return self.get_obs()
        
    def render(self):
        pass
        
    def get_reward(self, y):
        if self.done==True and np.argmax(self.q)==self.state:
            return 1
        elif y == 1:
            return -0.3
        else:
            return 0
       
         
    def step(self, action_label):
        act = act_space_map(action_label, 24)
        if action_label< 12:
            POVM, j = POVMsimple(act[0]/12), act[1]
        else:
            POVM, j = POVMtsimple((act[0]-12)/12), act[1]
        y=0
        if self.obs_state[j]==1:
            y=1
        self.take_action(POVM, j)
        self.obs_state[j] = int(1)
        self.done = isdone(self.obs_state, self.d)
        self.reward = self.get_reward(y)
        return self.get_obs(), self.reward, self.done, {}
        
    def take_action(self, POVM, j):
        pvec_matrix = prob_outcome(self.rho, POVM, j)
        out = outcome(self.state, pvec_matrix)
        self.q = PMstate(self.rho, POVM, self.q, j, out)
        self.done = isdone(self.obs_state, self.d)
