import math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import numpy as np

@dataclass 
class ForagingConfig:
    W:int=7
    patch_A:int=1
    patch_B:int=5
    base_reward:float=10.0
    decay_rate:float=0.6        # 改成几何衰减的参数
    reward_period:int=20
    gamma:float=0.5
    seed:int=42

class ForagingPlaygroundLinear:
    def __init__(self, cfg: ForagingConfig):
        self.cfg=cfg
        random.seed(cfg.seed); np.random.seed(cfg.seed)
        self.reset()

    def reset(self, start_pos: Optional[int]=None, start_t:int=0):
        self.t=start_t
        self.x=self.cfg.W//2 if start_pos is None else start_pos
        return self.x, self.t

    def current_active_patch(self, t:int)->int:
        ph = t % self.cfg.reward_period
        return self.cfg.patch_A if ph < self.cfg.reward_period/2 else self.cfg.patch_B

    def age_since_activation(self, t:int)->int:
        ph = t % self.cfg.reward_period
        return ph if ph < self.cfg.reward_period/2 else int(ph - self.cfg.reward_period/2)

    def true_reward(self, x:int, t:int)->float:
        if x != self.current_active_patch(t):
            return 0.0
        # ✅ 几何衰减：每步乘 decay_rate
        return self.cfg.base_reward * (self.cfg.decay_rate ** self.age_since_activation(t))

    def step(self, a:int)->Tuple[int,float,int]:
        self.x = max(0, min(self.cfg.W-1, self.x+int(np.clip(a,-1,1))))
        self.t += 1
        return self.x, self.true_reward(self.x,self.t), self.t


def _step_toward(s: int, target: int) -> int:
    # move one step toward target (respect edges)
    if s < target:
        s_next = s + 1
    elif s > target:
        s_next = s - 1
    else:
        s_next = s
    return s_next 

def _target_for_time(T: int) -> int:
    r = T % 20
    if 0 <= r <= 5:     # move/stay at 1 until r==6
        return 1
    if 6 <= r <= 15:    # move/stay at 5 through r==15
        return 5
    return 1

def reward_simulate(state, t,
                    base_reward: float = 10.0,
                    decay_rate: float = 0.6,
                    reward_period: int = 20):
    """
    """
    state = np.asarray(state).astype(int)
    t = np.asarray(t).astype(int)

    phase = t % reward_period
    half = reward_period // 2

    reward_A = np.where(
        (state == 1) & (phase < half),
        base_reward * (decay_rate ** phase),
        0.0
    )

    reward_B = np.where(
        (state == 5) & (phase >= half),
        base_reward * (decay_rate ** (phase - half)),
        0.0
    )

    reward = reward_A + reward_B

    if reward.shape == ():   # 0-d array
        return float(reward)
    return reward

def path_opt(state: int, t: int, len_future: int,
             base_reward: float = 10.0,
             decay_rate: float = 0.6,
             reward_period: int = 20):
    s = state
    path = []
    rewards = []
    for k in range(1, len_future+1):
        T = t + k
        target = _target_for_time(T-1)
        s = _step_toward(s, target)   # ✅ 每次只动一步
        path.append(s)
        rewards.append(
            reward_simulate(s, T, base_reward, decay_rate, reward_period)
        )
    return path, rewards

def compute_normalized_future_rewards(rewards, T, gamma):
    Y = np.zeros(T, dtype=float)
    gammas = gamma ** np.arange(T)  # [γ^0, γ^1, …, γ^{T-1}]

    for t in range(T):
        L = T - (t + 1)
        if L <= 0:
            continue
        raw = gammas[:L]
        weights = raw
        future_rewards = rewards[t+1 : t+1+L]
        Y[t] = np.dot(weights, future_rewards)

    return Y

def _successors(s: int):
    if s <= 0:  return (0, 1)
    if s >= 6:  return (5, 6)
    return (s-1, s, s+1)


### Encoding strategy
@dataclass
class EncodeConfig:
    W:int
    reward_period:int
    fourier_K:int=3
    @property
    def dim(self)->int: return self.W + self.reward_period + 2*self.fourier_K

class Encoder:
    def __init__(self, cfg: EncodeConfig):
        self.cfg=cfg
        W,P,K=cfg.W,cfg.reward_period,cfg.fourier_K
        self.Epos=np.eye(W,dtype=np.float32)
        self.Etime=np.eye(P,dtype=np.float32)
        F=np.zeros((P,2*K),dtype=np.float32)
        for ph in range(P):
            j=0
            for k in range(1,K+1):
                ang=2.0*math.pi*k*(ph/P)
                F[ph,j]=math.sin(ang); j+=1
                F[ph,j]=math.cos(ang); j+=1
        self.F=F

    def encode(self,x:int,t:int)->np.ndarray:
        ph=t%self.cfg.reward_period
        return np.concatenate((self.Epos[x], self.Etime[ph], self.F[ph]), axis=0)

    def encode_table(self, t_list: List[int])->np.ndarray:
        feats=[]
        for t in t_list:
            ph=t%self.cfg.reward_period
            block=np.concatenate((self.Epos,
                                  np.repeat(self.Etime[None,ph,:], self.cfg.W, axis=0),
                                  np.repeat(self.F[None,ph,:],   self.cfg.W, axis=0)),
                                 axis=1)  # [W, dim]
            feats.append(block.astype(np.float32))
        return np.stack(feats, axis=0)  # [T, W, dim]