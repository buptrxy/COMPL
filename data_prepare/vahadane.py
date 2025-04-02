import spams
import numpy as np
import cv2
import time


class vahadane(object):
    """
    Basic Self-Attention Module
    Args:
        STAIN_NUM (int): Number of stains to be separated.
        THRESH (float): Threshold value for stain separation.
        LAMBDA1 (float): Regularization parameter for dictionary learning.
        LAMBDA2 (float): Regularization parameter for sparse coding.
        ITER (int): Number of iterations for the dictionary learning process.
        fast_mode (int): Mode selection for speed; 0 for normal, 1 for fast.
        getH_mode (int): Mode for H computation; 0 for using spams.lasso, 1 for using pseudo-inverse.
    """

    def __init__(self, STAIN_NUM=2, THRESH=0.9, LAMBDA1=0.01, LAMBDA2=0.01, ITER=100, fast_mode=0, getH_mode=0):
        """
        Initializes the vahadane object with specified parameters.
        """
        self.STAIN_NUM = STAIN_NUM
        self.THRESH = THRESH
        self.LAMBDA1 = LAMBDA1
        self.LAMBDA2 = LAMBDA2
        self.ITER = ITER
        self.fast_mode = fast_mode  # 0: normal; 1: fast
        self.getH_mode = getH_mode  # 0: spams.lasso; 1: pinv;

    def show_config(self):
        """
        Prints the configuration parameters of the vahadane instance.
        """
        print('STAIN_NUM =', self.STAIN_NUM)
        print('THRESH =', self.THRESH)
        print('LAMBDA1 =', self.LAMBDA1)
        print('LAMBDA2 =', self.LAMBDA2)
        print('ITER =', self.ITER)
        print('fast_mode =', self.fast_mode)
        print('getH_mode =', self.getH_mode)

    def getV(self, img):
        """
        Computes the log-transformed matrix V from the input image.
        
        Args:
            img (ndarray): Input image in RGB format.
        
        Returns:
            tuple: A tuple containing two matrices, V0 (global values) and V (local values).
        """
        I0 = img.reshape((-1, 3)).T
        I0[I0 == 0] = 1
        V0 = np.log(255 / I0)

        img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        mask = img_LAB[:, :, 0] / 255 < self.THRESH
        I = img[mask].reshape((-1, 3)).T
        I[I == 0] = 1
        V = np.log(255 / I)

        return V0, V

    def getW(self, V):
        """
        Learns the dictionary W using the DL algorithm.
        
        Args:
            V (ndarray): The input matrix for dictionary learning.
        
        Returns:
            ndarray: The learned dictionary W, normalized.
        """
        W = spams.trainDL(np.asfortranarray(V), numThreads=1, K=self.STAIN_NUM, lambda1=self.LAMBDA1, iter=self.ITER, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False)
        W = W / np.linalg.norm(W, axis=0)[None, :]
        if (W[0, 0] < W[0, 1]):
            W = W[:, [1, 0]]
        return W

    def getH(self, V, W):
        """
        Computes the sparse coding H for given V and W matrices.
        
        Args:
            V (ndarray): The input matrix.
            W (ndarray): The dictionary matrix.
        
        Returns:
            ndarray: The computed sparse matrix H.
        """
        if (self.getH_mode == 0):
            H = spams.lasso(np.asfortranarray(V), np.asfortranarray(W), numThreads=1, mode=2, lambda1=self.LAMBDA2, pos=True, verbose=False).toarray()
        elif (self.getH_mode == 1):
            H = np.linalg.pinv(W).dot(V)
            H[H < 0] = 0
        else:
            H = 0
        return H

    def stain_separate(self, img):
        """
        Separates the stains in the input image using the specified methods and parameters.
        
        Args:
            img (ndarray): Input image in RGB format.
        
        Returns:
            tuple: A tuple containing the dictionary W and sparse coding H for the image.
        """
        start = time.time()
        if (self.fast_mode == 0):
            V0, V = self.getV(img)
            W = self.getW(V)
            H = self.getH(V0, W)
        elif (self.fast_mode == 1):
            m = img.shape[0]
            n = img.shape[1]
            grid_size_m = int(m / 5)
            lenm = int(m / 20)
            grid_size_n = int(n / 5)
            lenn = int(n / 20)
            W = np.zeros((81, 3, self.STAIN_NUM)).astype(np.float64)
            for i in range(0, 4):
                for j in range(0, 4):
                    px = (i + 1) * grid_size_m
                    py = (j + 1) * grid_size_n
                    patch = img[px - lenm: px + lenm, py - lenn: py + lenn, :]
                    V0, V = self.getV(patch)
                    W[i * 9 + j] = self.getW(V)
            W = np.mean(W, axis=0)
            V0, V = self.getV(img)
            H = self.getH(V0, W)
        return W, H

    def SPCN(self, img, Ws, Hs, Wt, Ht):
        """
        Performs the stain color normalization process on the input image.
        
        Args:
            img (ndarray): The input image.
            Ws (ndarray): The source stain dictionary.
            Hs (ndarray): The source sparse coding matrix.
            Wt (ndarray): The target stain dictionary.
            Ht (ndarray): The target sparse coding matrix.
        
        Returns:
            ndarray: The normalized image.
        """
        Hs_RM = np.percentile(Hs, 99)
        Ht_RM = np.percentile(Ht, 99)
        Hs_norm = Hs * Ht_RM / Hs_RM
        Vs_norm = np.dot(Wt, Hs_norm)
        Is_norm = 255 * np.exp(-1 * Vs_norm)
        I = Is_norm.T.reshape(img.shape).astype(np.uint8)
        return I
