from build_matrix import build_matrix
import os
from scipy.sparse import load_npz
from scipy.sparse.linalg import lsqr, lsmr
import numpy as np
from scipy.sparse import vstack,  identity
from pathlib import Path


class recovery:
    def __init__(self):
        base = Path(__file__).parent

        self.config = base / "Config Files" / "GR150R.F150W.220725.conf"
        self.wave = base / "jwst_niriss_wavelengthrange_0002.asdf"
        self.filter = "F150W"
        
        
        if os.path.exists("H_matrix_F150W_flux_20_500_orders_PCA_sensitivity.npz"): #checks if file exists
            self.H_PCA_sens = load_npz("H_matrix_F150W_flux_20_500_orders_PCA_sensitivity.npz") #loads stored traces matrix
           
        else:
            self.H_PCA_sens = None
            
        

    
    def recover_direct_from_traces_basis_matrix_PCA(self, dispersed, image=True, initial_guess=None):
        """Function to recover direct image from GIVEN IMAGE dispersed. Uses the precomputed traces matrix H to recover the direct image from a dispersed image 
        via least squares."""
        H = self.H_PCA_sens[:, initial_guess == 1] # trims the matrix to just its possible source columns
      
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(H,f, iter_lim=500, show=True) #solves min_d ||Ad-f||^2.
        d = result[0] #  lsqr stores result as final_solution, istop, itn.... So we use only [0]
        
        # recovers full d with zeros at the correct positions
        x = np.zeros_like(initial_guess, dtype=float)
        x[initial_guess==1]=d
        d= x
        
        if image == False: # If we want coefficient vector d
            return d
        
        # else, we want ready visualized image
        A = build_matrix(self.config, filter_name=self.filter, wavelengthrange_file=self.wave)
        
        Recovered = A.integrated_flux_image_PCA(d)
     
        return Recovered


    def recover_direct_from_traces_basis_matrix_PCA_thikonov_variance(self, dispersed, lam=1e-2):
        """
        Recover direct image from dispersed image using:

        min_{d >= 0} ||W(Hd - f)||^2 + lambda ||d||^2

        where:
            W = diag(1/sigma)
            sigma estimated from data (Poisson + read noise)
        """
        print("Variance")

        # 1. Flatten image

        print("Hi")
        f = dispersed.astype(float).ravel()


        # 2. Estimate sigma (noise model)
        #    sigma^2 = f + read_noise^2
        
        read_noise = 5  # reasonable default (can tune)
    
        variance = np.maximum(f,0) + read_noise**2
        sigma = np.sqrt(variance)  
        print("Hii")
        sigma_inv = 1.0 / sigma  # precision vector of sqrt, so sigma^-1/2

        H_sigma = self.H_PCA_sens.multiply(sigma_inv[:,None])
        f_sigma = sigma_inv*f
        print("Hiii")
        # =====================================================
        # 4. Tikhonov regularization
        #    augment system:
        #    [H_w        ] d ≈ [f_w]
        #    [√λ I       ]     [0  ]

        N= H_sigma.shape[1]
        
        H_reg = vstack([H_sigma, np.sqrt(lam)*identity(N)])
        print("Hiiii")
        f_reg = np.concatenate([f_sigma,np.zeros(N)])
        print("Hiiiii")
        
        # 5. Solve nonnegative least squares
       

        res = lsmr(H_reg, f_reg, atol=1e-15, btol=1e-15, maxiter=1000, show= True)
        d = res[0]


        print("Hiiiiii")
    
        # 6. Reconstruct image

        A = build_matrix(self.config, filter_name=self.filter, wavelengthrange_file=self.wave)
        Recovered = A.integrated_flux_image_PCA(d)

        return Recovered
    
    

