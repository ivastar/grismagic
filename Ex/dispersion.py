import os
from scipy.sparse import load_npz
import numpy as np

class dispersion:
    def __init__(self):
        """Loads matrix for linear dispersion"""
            
        if os.path.exists("H_matrix_F150W_flux_20_500_orders_PCA_sensitivity.npz"): #checks if file exists
            self.H_PCA_sens = load_npz("H_matrix_F150W_flux_20_500_orders_PCA_sensitivity.npz") #loads stored traces matrix
           
        else:
            self.H_PCA_sens = None

    
    def dispersed_PCA(self,a_tilde):
        """Computes H@a_tilde = f, for a_tilde containing the coefficients a_1,...,a_k for each pixel"""
        f = self.H_PCA_sens@a_tilde
        dispersed = f.reshape(500,20)
        return dispersed