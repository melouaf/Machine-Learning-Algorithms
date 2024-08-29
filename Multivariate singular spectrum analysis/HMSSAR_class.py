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


class HMSSAR:
    def embed(self, X: List[np.array]) -> (np.array, np.array):
        '''
        Performs embedding step.
        Inputs:
          X: a list of time-series.
        Returns:
          Xh: a horizontally concatenated list of trajectory matrices.
          Ki: the list of lengths used for trajectory matrices computation.
        '''
        return np.hstack([sliding_window_view(x=x, window_shape=self.L, axis=0).squeeze().T for x in X]), np.array([len(x) - self.L + 1 for x in X])

    def group(self, Xh: np.array) -> np.array:
        '''
        Performs grouping step.
        Inputs:
          Xh: a horizontally concatenated list of trajectory matrices.
        Returns:
          Xrec: a reconstructed trajectory matrix using SVD.
        '''
        Xrec = np.zeros_like(Xh)
        for i in range(self.r):
            Xrec += self.Lambda[i] * self.U[:,i][:, None] @ self.V[i,:][:, None].T
        return Xrec

    def hankelize(self, Xrec: np.array) -> List[np.array]:
        '''
        Performs reconstruction step.
        Inputs:
          Xrec: a reconstructed trajectory matrix using SVD.
        Returns:
          Xtilde: a list of reconstructed time-series using diagonal averaging (Hankelization).
        '''
        Xtilde = []
        for i in range(self.M):
            clear_output(True)
            print(f"Processed serie: {i+1}/{self.M}")
            Xrec_i = Xrec[:, self.onsets[i]:self.onsets[i+1]]
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
        self.L = round(self.M * (self.N + 1.0) / (self.M + 1.0))

        clear_output(True)
        print("Embedding ...")

        Xh, self.Ki = self.embed(X)

        self.Ni = self.L + self.Ki - 1
        self.onsets = [0] + list(np.cumsum(self.Ki))
        self.offsets = list(np.cumsum(self.Ki) - 1)

        self.U, self.Lambda, self.V = np.linalg.svd(Xh, full_matrices=False)

        # Sanity check:
        if sanity_check:
            Xrec = np.zeros_like(Xh)
            for i in range(min(Xh.shape[0], self.r)):
                Xrec += self.Lambda[i] * self.U[:,i][:, None] @ self.V[i,:][:, None].T
            assert np.allclose(Xh, Xrec), "SVD decomposition didn't yield the desired result."

        clear_output(True)
        print("Grouping ...")

        Xrec = self.group(Xh)

        clear_output(True)
        print("Reconstructing ...")

        Xtilde = self.hankelize(Xrec)

        Pi = self.U[-1,:self.r][:, None]
        Unabla = self.U[:-1,:self.r]
        v = np.linalg.norm(Pi)
        assert v < 1, "v >= 1, HMSSAR can't be performed."
        self.R = (1.0 / (1 - v**2)) * (Unabla @ Pi)
        self.Zh = np.concatenate([Xtilde[i][-self.L+1:][:, None] for i in range(self.M)], axis=1)

        clear_output(True)
        print("Data fitted.")

    def plot_eigs(self, Lambda_max: int = 10) -> None:
        '''
        Plots the SVD-resulting eigenvalues.
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
            y = self.R.T @ Zh
            pred.append(y.T)
            Zh = np.concatenate([Zh[1:,:], y], axis=0)
        pred = np.concatenate(pred, axis=1)
        return pred


