import math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class ForagingConfig:
    W:int=7
    patch_A:int=1
    patch_B:int=5
    base_reward:float=10.0
    decay_rate:float=0.6
    reward_period:int=20         
    seed=42


class ForagingPlaygroundLinear:
    """ Linear Foraging Environment"""
    def __init__(self, cfg: ForagingConfig):
        self.cfg=cfg
        random.seed(cfg.seed)
        self.reset()

    def reset(self, start_pos: Optional[int]=None, start_t:int=0):
        self.t=start_t
        self.x=self.cfg.W//2 if start_pos is None else start_pos
        return self.x, self.t

    def current_active_patch(self, t:int)->int:
        phase = t % self.cfg.reward_period        # 0..19
        return self.cfg.patch_A if phase < 10 else self.cfg.patch_B

    def time_since_activation(self, t:int)->int:
        phase = t % self.cfg.reward_period        # 0..19
        return phase if phase < 10 else (phase - 10)
    
    def true_reward(self, x:int, t:int)->float:
        if x != self.current_active_patch(t):
            return 0.0
        age = self.time_since_activation(t)
        return self.cfg.base_reward * (self.cfg.decay_rate ** age)

    def step(self, a:int)->Tuple[int,float,int]:
        self.x=max(0,min(self.cfg.W-1,self.x+int(np.clip(a,-1,1))))
        self.t+=1
        return self.x, self.true_reward(self.x,self.t), self.t


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