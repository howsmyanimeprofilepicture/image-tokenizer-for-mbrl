from typing import NamedTuple, List
import numpy as np
import pickle


class TrajSpan(NamedTuple):
    obs: List[np.ndarray]
    action: List[int]
    reward: List[float]
    done: List[bool]


class Trajectory:
    """Trajectory class to collect offline dataset"""

    @staticmethod
    def from_pickle(file_path):
        """Args
        - file_path(str): the path of pickle file
        """
        with open(file_path, "rb") as f:
            traj: Trajectory = pickle.load(f)
        return traj

    def __init__(self) -> None:

        self.obs: List[np.ndarray] = []
        self.action: List[int] = []
        self.reward: List[float] = []
        self.done: List[bool] = []

    def __getitem__(self, key):
        if type(key) == int:
            key = slice(key, key+1)
        return TrajSpan(self.obs[key],
                        self.action[key],
                        self.reward[key],
                        self.done[key])

    def add(self, obs, action, reward, done):
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
