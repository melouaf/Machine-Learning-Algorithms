import numpy as np
import matplotlib.pyplot as plt
from typing import List
from numpy.lib.stride_tricks import sliding_window_view
from IPython.display import clear_output


def average_anti_diag(traj_matrix: np.ndarray) -> np.ndarray:
    """Average anti diagonal elements of a 2d array"""
    x1d = [
        np.mean(traj_matrix[::-1, :].diagonal(i))
        for i in range(-traj_matrix.shape[0] + 1, traj_matrix.shape[1])
    ]
    return np.array(x1d)


def average_anti_diag(traj_matrix: np.ndarray) -> np.ndarray:
    """Average anti diagonal elements of a 2d array"""
    x1d = [
        np.mean(traj_matrix[::-1, :].diagonal(i))
        for i in range(-traj_matrix.shape[0] + 1, traj_matrix.shape[1])
    ]
    return np.array(x1d)

class VMSSAR:
    def embed(self, X: List[np.array]) -> np.array:
        '''
        Performs embedding step.
        Inputs:
          X: a list of time-series.
        Returns:
          Xv: a vertically concatenated list of trajectory matrices.
          Li: the list of lengths used for trajectory matrices computation.
        '''
        return np.vstack([sliding_window_view(x=x, window_shape=len(x)-self.K+1, axis=0).squeeze().T for x in X]), np.array([len(x)-self.K+1 for x in X])

    def group(self, Xv: np.array) -> np.array:
        '''
        Performs grouping step.
        Inputs:
          Xv: a vertically concatenated list of trajectory matrices.
        Returns:
          Xrec: a list of reconstructed trajectory matrices using SVD.
        '''
        Xrec = np.zeros_like(Xv)
        for i in range(self.r):
            Xrec += self.Lambda[i]*self.U[:,i][:,None] @ self.V[i,:][:,None].T
        return Xrec

    def hankelize(self, Xrec: np.array) -> List[np.array]:
        '''
        Performs reconstruction step.
        Inputs:
          Xrec: a list of reconstructed trajectory matrices using SVD.
        Returns:
          Xtilde: a list of reconstructed time-series using diagonal averaging (Hankelization).
        '''
        Xtilde = []
        for i in range(self.M):
            print(f"Processed serie: {i+1}/{self.M}")
            clear_output(True)
            Xrec_i = Xrec[self.onsets[i]:self.onsets[i+1],:]
            Xtilde.append(average_anti_diag(Xrec_i))
        return Xtilde

    def fit(self, X: List[np.array], r: int = 6, sanity_check: bool = False) -> None:
        '''
        Fits the training time-series.
        Inputs:
          X: a list of time-series.
        Returns:
        '''
        self.r = r
        self.M = len(X)
        self.N = min(map(len, X))
        self.K = round(self.M*(self.N+1.0)/(self.M+1.0))

        clear_output(True)
        print("Embedding ...")

        Xv, self.Li = self.embed(X)

        self.Ni = self.Li + self.K - 1
        self.onsets = [0] + list(np.cumsum(self.Li))
        self.offsets = list(np.cumsum(self.Li)-1)

        self.U, self.Lambda, self.V = np.linalg.svd(Xv, full_matrices=False)

        # Sanity check:
        if sanity_check:
            Xrec = np.zeros_like(Xv)
            for i in range(min(Xv.shape)):
                Xrec += self.Lambda[i]*self.U[:,i][:,None] @ self.V[i,:][:,None].T
            assert np.allclose(Xv, Xrec), "SVD decomposition didn't yield the desired result."

        clear_output(True)
        print("Grouping ...")

        Xrec = self.group(Xv)

        clear_output(True)
        print("Reconstructing ...")

        Xtilde = self.hankelize(Xrec)

        s = np.array([(i in self.offsets) for i in range(self.U.shape[0])])
        self.W = self.U[s, :self.r]
        self.Unabla = self.U[~s, :self.r]
        self.Zh = np.concatenate([Xtilde[i][-self.Li[i]+1:][:, None] for i in range(self.M)], axis=0)

        clear_output(True)
        print("Data fitted.")

    def plot_eigs(self, Lambda_max: int = 10) -> None:
        '''
        Plots the svd-resulting eigenvalues.
        '''
        plt.plot(self.Lambda[:Lambda_max], "-*")
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title('SVD Eigenvalues')
        plt.show()

    def forecast(self, h: int = 1000) -> np.array:
        '''
        Forecasts the fitted time-series.
        Inputs:
          h: forecasting horizon.
        Outputs:
          pred: forecasts for the fitted time-series.
        '''
        Zh = self.Zh
        pred = []
        for _ in range(h):
            y = ((np.linalg.inv(np.eye(self.M)-self.W @ self.W.T) @ self.W) @ (self.Unabla.T)) @ Zh
            pred.append(y)
            Zh = rollAndAdd(Zh, y, self.Li-1)
        pred = np.concatenate(pred, axis=1)
        return pred


