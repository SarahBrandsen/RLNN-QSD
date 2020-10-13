import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer as Trainer
import ray.rllib.agents.ppo.ppo as ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

print(ray.__version__)
import itertools
import numpy as np
import copy
import matplotlib.pyplot as plt
import tikzplotlib as tikz
import scipy

import qsd_library_multiple as qsdm
import qsd_library_last as qsdl
import qsd_library_copy as qsdc
import sdp

# Common parameters for all runs
num_gpus = 0.25
num_workers = 4
ray_init_kwargs = {"object_store_memory": 5*1024**3, "num_cpus": num_workers+1}

###local SDP

def PMstate(rho, POVM, q, outcome):
    denom = 0
    for k in range(len(rho)):
        denom += q[k]*np.trace(POVM[outcome]@rho[k])
    for k in range(len(rho)):
        q[k] = q[k]*np.trace(POVM[outcome]@rho[k])/denom
    return q

    
def POVMsimple(x):
    return np.array([[[np.cos(np.pi*x/2)**2, np.cos(np.pi*x/2)*np.sin(np.pi*x/2)],
                      [np.cos(np.pi*x/2)*np.sin(np.pi*x/2), np.sin(np.pi*x/2)**2]],
                     [[np.sin(np.pi*x/2)**2, -np.cos(np.pi*x/2)*np.sin(np.pi*x/2)],
                      [-np.cos(np.pi*x/2)*np.sin(np.pi*x/2), np.cos(np.pi*x/2)**2]]])


def expected_sdp(pvec, rho, nrem, POVM):
    rhoav = np.zeros((2, 2))
    for _ in range(len(pvec)):
        rhoav += pvec[_]*rho[_]
    prob0 = np.trace(rhoav@POVM[0])
    prob1 = np.trace(rhoav@POVM[1])
    pvec_0 = PMstate(rho, POVM, copy.copy(pvec), 0)
    pvec_1 = PMstate(rho, POVM, copy.copy(pvec), 1)
    if nrem>1:
        fullrho = np.array([[rho[_] for k in range(nrem-1)] for _ in range(len(rho))])
        return prob0*sdp.SDP(fullrho, pvec_0, nrem-1)+prob1*sdp.SDP(fullrho, pvec_1, nrem-1)
    else:
        return prob0*np.max(pvec_0)+prob1*np.max(pvec_1) 
    
    
def max_SDP_POVM(pvec, rho, nrem):
    vec = np.array([expected_sdp(pvec, rho, nrem, POVMsimple(_/20)) for _ in range(20)])
    return np.argmax(vec)

def max_SDP_sim(pvec, rho, N, nrounds, d):
    defaultconfig = {"d": d, "rho": rho, "q": pvec, 
                    "separable": True, "quantization": 20, "meas_quant": np.array([1,1,1,1])}
    reward = 0
    for k in range(nrounds):
        env = qsdc.QSDEnv(defaultconfig)
        env.reset()
        for j in range(N-1):
            q = copy.copy(env.q)
            act = max_SDP_POVM(q, rho, N-j)
            _, r, _, _ = env.step(act)
        fullrho = np.array([[rho[_] for k in range(1)] for _ in range(len(rho))])
        reward += sdp.SDP(fullrho, env.q, 1)
        if k%50 == 0:
            print(reward/(k+1))
    return reward/nrounds

def expected_sdp_order(pvec, rhoind, nrem, fullrho, POVM):
    rhoav = np.zeros((2, 2))
    for _ in range(len(pvec)):
        rhoav += pvec[_]*rhoind[_]
    prob0 = np.trace(rhoav@POVM[0])
    prob1 = np.trace(rhoav@POVM[1])
    pvec_0 = PMstate(rhoind, POVM, copy.copy(pvec), 0)
    pvec_1 = PMstate(rhoind, POVM, copy.copy(pvec), 1)
    if nrem>1:
        return prob0*sdp.SDP(fullrho, pvec_0, nrem-1)+prob1*sdp.SDP(fullrho, pvec_1, nrem-1)
    else:
        return prob0*np.max(pvec_0)+prob1*np.max(pvec_1) 
    

    

def max_SDP_POVM_order(pvec, rho, jrem):
    jopt = jrem[0]
    actopt = 0
    valopt = 0
    for j in jrem:
        rhoind = np.array([rho[_][j] for _ in range(len(rho))])
        index = np.argwhere(jrem==j)
        jremp = np.delete(copy.copy(jrem), index)
        fullrho = np.array([[rho[_][k] for k in jremp] for _ in range(len(rho))])
        vec = np.array([expected_sdp_order(pvec, rhoind, len(jrem), fullrho, POVMsimple(_/20)) for _ in range(20)])
        if np.argmax(vec)>valopt:
            valopt = np.max(vec)
            jopt = j
            actopt = np.argmax(vec)
    index = np.argwhere(jrem==jopt)
    jrem = np.delete(jrem, index)
    return jopt, actopt, jrem

def max_SDP_sim_order(pvec, rho, N, nrounds, d):
    defaultconfig = {"d": d, "rho": rho, "q": pvec, 
                    "separable": True, "quantization": 20}
    jrem = np.arange(len(rho[0]))
    jopt0, actopt0, jrem0 = max_SDP_POVM_order(pvec, rho, jrem)
    reward = 0
    for k in range(nrounds):
        jrem = copy.copy(jrem0)
        if k%50 == 0:
            print(reward/(k+1))
        env = qsdl.QSDEnv(defaultconfig)
        env.reset()
        act = jopt0*20+actopt0
        env.step(act)
        for j in range(N-1):
            q = copy.copy(env.q)
            jopt, actopt, jrem = max_SDP_POVM_order(q, rho, jrem)
            act = jopt*20+actopt
            _, r, _, _ = env.step(act)
            reward += r
    return reward/nrounds





###########################

def probsucc_LG(pvec, rhoind, POVM):
    rhoav = np.zeros((2, 2))
    for _ in range(len(pvec)):
        rhoav += pvec[_]*rhoind[_]
    prob0 = np.trace(rhoav@POVM[0])
    prob1 = np.trace(rhoav@POVM[1])
    pvec_0 = PMstate(rhoind, POVM, copy.copy(pvec), 0)
    pvec_1 = PMstate(rhoind, POVM, copy.copy(pvec), 1)
    return prob0*np.max(pvec_0)+prob1*np.max(pvec_1) 

def max_action_LG(pvec, rho, j):
    rhoind = np.array([rho[_][j] for _ in range(len(rho))])
    vec = np.array([probsucc_LG(pvec, rhoind, POVMsimple(_/20)) for _ in range(20)])
    return np.argmax(vec)

def LG_sim_order(pvec, rho, N, nrounds, d):
    defaultconfig = {"d": d, "rho": rho, "q": pvec, "separable": True, "quantization": 20}
    reward = 0
    for k in range(nrounds):
        if k%50 == 0:
            print(reward/(k+1))
        env = qsdl.QSDEnv(defaultconfig)
        env.reset()
        for j in range(N-1):
            q = copy.copy(env.q)
            actopt = max_action_LG(q, rho, j)
            act = j*20+actopt
            _, r, _, _ = env.step(act)
            reward += r
    return reward/nrounds

############# General function for testing


def single_test(defaultconfig, training_trials, evaluation_trials, check, lr = 0.00005, num_workers = 4, num_gpus = 0.25):
    ray.shutdown()
    ray.init(**ray_init_kwargs)
    config = ppo.DEFAULT_CONFIG.copy()
    if (num_gpus > 0):
        config["num_gpus"] = num_gpus
    config["num_workers"] = num_workers
    config["lr"] = lr
    config["train_batch_size"] = 8000
    config["num_sgd_iter"] = 5
    config["env_config"] = defaultconfig
    trainer = Trainer(config=config, env=qsdl.QSDEnv)
    for i in range(training_trials):
        result = trainer.train()
        print("train iteration",i+1,"/",training_trials," avg_reward =", 
              result["episode_reward_mean"]," timesteps =", result["timesteps_total"])
        if i % check == check-1:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
    avgR = 0
    for i in range(evaluation_trials):
        env=qsdl.QSDEnv(defaultconfig)
        obs = env.reset()
        done = False
        while not done:
            action = trainer.compute_action(obs)
            obs, r, done, _ = env.step(action)
            avgR += r
    return avgR/evaluation_trials


# General function for testing
def fulltest(total_trials, training_trials, d, m, q, train_check, evaluation_trials = 5000, lr = 0.00005, num_workers = 4, num_gpus = 0.25, SDP = True, LG = False, local_SDP = False, dep = True, rngvec = np.ones(1000)):
    quantization = 20
    separable = True
    bigvec = np.zeros((total_trials, int(training_trials/train_check)+1))
    vec_SDP = []
    vec_local_SDP = []
    vec_LG = []
    
    for j in range(total_trials):
        print("Starting round",j,"of",total_trials)
        rho, _ = qsdl.generate_initial_state(d, m, rng = rngvec[j], depolarized = dep)
        
        if local_SDP == True:
            lg = max_SDP_sim_order(q, rho, len(d), 1250, d)
            vec_local_SDP.append(lg)
            print("local SDP-based")
            print(lg)
        if SDP == True:
            sdpr = sdp.SDP(rho, q, len(d))
            vec_SDP.append(sdpr)
            print("SDP")
            print(sdpr)
        if LG == True:
            lg = LG_sim_order(copy.copy(q), copy.copy(rho), len(d), 2500, d)
            vec_LG.append(lg)
            print("LG")
            print(lg)
        
        print("RLNN: ")
        print(bigvec[-1])
        defaultconfig = {"rho": copy.copy(rho), "q": copy.copy(q),  
                 "quantization" : quantization, "d" : d, "separable": True}
        vec = []
        ray.shutdown()
        ray.init(**ray_init_kwargs)
        config = ppo.DEFAULT_CONFIG.copy()
        if (num_gpus > 0):
            config["num_gpus"] = num_gpus
            config["num_workers"] = num_workers
            config["lr"] = lr
            config["train_batch_size"] = 8000
            config["num_sgd_iter"] = 5
            config["env_config"] = defaultconfig
            trainer = Trainer(config=config, env=qsdl.QSDEnv)
        for i in range(training_trials):
            result = trainer.train()
            print("train iteration",i+1,"/",training_trials," avg_reward =", 
              result["episode_reward_mean"]," timesteps =", result["timesteps_total"])
    #         if i % check == check-1:
    #             checkpoint = trainer.save()
    #             print("checkpoint saved at", checkpoint)
            if i == 0 or (i+1) % train_check == 0:
                rew = 0
                for i in range(evaluation_trials):
                    env=qsdl.QSDEnv(defaultconfig)
                    obs = env.reset()
                    done = False
                    while not done:
                        action = trainer.compute_action(obs)
                        obs, r, done, _ = env.step(action)
                        rew += r
                vec.append(rew/evaluation_trials)       
        bigvec[j] = vec
    return bigvec, vec_SDP, vec_local_SDP, vec_LG

