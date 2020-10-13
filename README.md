# RLNN-QSD

We develop a reinforcement learning with neural networks (RLNN) approach for m-ary quantum state discrimination. This algorithm can be applied to any set of real, tensor product quantum candidate states. Results are generated using the default PPO algorithm from the RLlib package included in Ray version 0.7.3. 

#### How to Use

```from qsd_library_last import *
from sdp import *
import ray
from ray import tune
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer as Trainer


#number of training iterations for the neural network
training_trials = 200
#number of evaluation episodes after training
evaluation_trials = 5000
#number of training iterations between neural network checkpoint
check = 50

#dimension vector of quantum subsystems (eg [2,2,2] corresponds to 3 qubit susbystems, [2,2,2,2] corresponds to 4 qubit subsystems, etc)
d = np.array([2,2,2])
#number of candidate states
m = 3
#quantization of action space, determines number of total quantum measurements agent chooses from
quantization = 20
#leave this parameter fixed
separable = True
#starting prior vector for the candidate states
q = np.array([1/3, 1/3, 1/3])

# generate candidate states. Change third argument to "False" to remove depolarizing noise
rho, _ = generate_initial_state(d, m, True)
defaultconfig = {"rho": copy.copy(rho), "q": copy.copy(q),  
                 "quantization" : quantization, "d" : d, "separable": True}

# train neural network and evaluate resulting success probability
ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0.25
config["num_workers"] = 5
config["lr"] = 0.00005
config["num_sgd_iter"] = 15
config["env_config"] = defaultconfig
trainer = Trainer(config=config, env=QSDEnv)

#training
for i in range(training_trials):
  result = trainer.train()
  print("train iteration",i+1,"/",training_trials," avg_reward =", 
  result["episode_reward_mean"]," timesteps =", result["timesteps_total"])
  if i % check == check-1:
    checkpoint = trainer.save()
  avgR = 0

#evaluation
for i in range(evaluation_trials):
    env=QSDEnv(defaultconfig)
    obs = env.reset()
    done = False
    while not done:
        action = trainer.compute_action(obs)
        obs, r, done, _ = env.step(action)
        avgR += r
#final success probability for NN
NN_succ_prob = avgR/evaluation_trials

#find optimal collective success probability
SDP_succ_prob = SDP(rho, q, len(d))

print("Success probability with RLNN:")
print(NN_succ_prob)
print("Success probability with SDP:")
print(SDP_succ_prob)
```

