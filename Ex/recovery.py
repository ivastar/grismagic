from build_matrix import build_matrix
import os
from scipy.sparse import load_npz
from scipy.sparse.linalg import lsqr, lsmr
import numpy as np
from scipy.optimize import lsq_linear
from scipy.optimize import linprog
from scipy.linalg import pinv
import matplotlib.pyplot as plt
from scipy.sparse import vstack, diags, identity


class recovery:
    def __init__(self):
        if os.path.exists("A_matrix_with_trace_count.npz"): #checks if file exists
            self.A_full = load_npz("A_matrix_with_trace_count.npz") #loads stored traces matrix
            self.trace_count = self.A_full[-1].toarray().ravel() #gives the amount of trace pixels per column of A. A1 gives 1D vector
            self.A=self.A_full[:-1] #keeps all rows except the last one, so A is the trace build matrix again
        else:
            self.A_full = None
            self.A = None
            self.trace_count = None
            
        if os.path.exists("H_matrix_flux_gaussian_all_orders.npz"): #checks if file exists
            self.H_full = load_npz("H_matrix_flux_gaussian_all_orders.npz") #loads stored traces matrix
           
        else:
            self.H_full = None
            
        if os.path.exists("A_F150W_20_500_matrix_with_trace_count_sensitivities_all_orders.npz"): #checks if file exists
            self.ASens_full = load_npz("A_F150W_20_500_matrix_with_trace_count_sensitivities_all_orders.npz") #loads stored traces matrix
            self.trace_countSens = self.ASens_full[-1].toarray().ravel() #gives the amount of trace pixels per column of A. A1 gives 1D vector
            self.ASens=self.ASens_full[:-1] #keeps all rows except the last one, so A is the trace build matrix again
        else:
            self.ASens_full = None
            self.ASens = None
            self.trace_countSens = None
                
        if os.path.exists("H_matrix_flux_1st_order_PCA.npz"): #checks if file exists
            self.H_PCA = load_npz("H_matrix_flux_1st_order_PCA.npz") #loads stored traces matrix
           
        else:
            self.H_PCA = None
        
        if os.path.exists("H_matrix_F150W_flux_20_500_orders_PCA_sensitivity.npz"): #checks if file exists
            self.H_PCA_sens = load_npz("H_matrix_F150W_flux_20_500_orders_PCA_sensitivity.npz") #loads stored traces matrix
           
        else:
            self.H_PCA_sens = None
            
        
        
    def recover_direct_from_traces_matrix(self, dispersed):
        """Function to recover direct image from SELF-COMPUTED dispersed. Uses the precomputed traces matrix A to recover the direct image from a dispersed image 
        via least squares."""
        m,n = dispersed.shape
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(self.A,f) #solves min_d ||Ad-f||^2
        d_recovered = result[0]
        
        d=d_recovered*self.trace_count #recovers total intensity for uniform ditribution
        
        Recovered = d.reshape(m, n) #transforms lsqr solution to matrix
        #Recovered[Recovered<0.05]=0 #small values are background error so ignore this
        
        return Recovered
    
    def recover_via_lsqr_bounds(self, dispersed):
        """uniform dist"""
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
       
        result= lsq_linear(self.A, f, bounds=(0, np.inf))
        d=result.x*self.trace_count #recovers total intensity for uniform ditribution
        

        A = build_matrix()
        Recovered = A.integrated_flux_image(d)
        return Recovered
    
    def recover_direct_from_traces_basis_matrix(self, dispersed):
        """Function to recover direct image from GIVEN IMAGE dispersed. Uses the precomputed traces matrix H to recover the direct image from a dispersed image 
        via least squares."""
        print(self.H_full.shape)
        print(dispersed.shape)
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(self.H_full,f) #solves min_d ||Ad-f||^2.
        d = result[0] #  lsqr stores result as final_solution, istop, itn.... So we use only [0]
        
        A = build_matrix()
        Recovered = A.integrated_flux_image(d)
     
        return Recovered
    
    def recover_direct_from_traces_sensitivities_matrix(self, dispersed):
        """Function to recover direct image from SELF-COMPUTED dispersed. Uses the precomputed traces matrix A to recover the direct image from a dispersed image 
        via least squares."""
        m,n = dispersed.shape
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(self.ASens,f, iter_lim=500, show=True) #solves min_d ||Ad-f||^2
        d_recovered = result[0]
        
        #d=d_recovered*self.trace_count #recovers total intensity for uniform ditribution
        
        Recovered = d_recovered.reshape(m, n) #transforms lsqr solution to matrix
        #Recovered[Recovered<0.05]=0 #small values are background error so ignore this
        
        return Recovered
    
    def recover_direct_from_traces_basis_matrix_PCA(self, dispersed, image=True, initial_guess=None):
        """Function to recover direct image from GIVEN IMAGE dispersed. Uses the precomputed traces matrix H to recover the direct image from a dispersed image 
        via least squares."""
        H = self.H_PCA_sens[:, initial_guess == 1]
        m,n=dispersed.shape
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(H,f, iter_lim=500, show=True) #solves min_d ||Ad-f||^2.
        d = result[0] #  lsqr stores result as final_solution, istop, itn.... So we use only [0]
        ############################################
        x = np.zeros_like(initial_guess, dtype=float)
        x[initial_guess==1]=d
        d= x
        
        if image == False:
            return d
        
        A = build_matrix()
        
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
        # =====================================================
        # 1. Flatten image
        # =====================================================
        print("Hi")
        f = dispersed.astype(float).ravel()

        # =====================================================
        # 2. Estimate sigma (noise model)
        #    sigma^2 = f + read_noise^2
        # =====================================================
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
        # =====================================================
        N= H_sigma.shape[1]
        
        H_reg = vstack([H_sigma, np.sqrt(lam)*identity(N)])
        print("Hiiii")
        f_reg = np.concatenate([f_sigma,np.zeros(N)])
        print("Hiiiii")
        # 5. Solve nonnegative least squares
        # =====================================================

        res = lsmr(H_reg, f_reg, atol=1e-15, btol=1e-15, maxiter=1000, show= True)
        d = res[0]

        # enforce nonnegativity
        #d = np.maximum(d, 0)

        print("Hiiiiii")
        # =====================================================
        # 6. Reconstruct image
        # =====================================================
        A = build_matrix()
        Recovered = A.integrated_flux_image_PCA(d)

        return Recovered
    
    
    def recover_pseudoinverse(self, dispersed):
        print("pseudoinverse")
        pseudo_H= pinv(self.H_PCA_sens.toarray()) # moore penrose pseudoinverse
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        d = pseudo_H@f
        A = build_matrix()
        Recovered = A.integrated_flux_image_PCA(d)
        return Recovered
    
    def recover_clip(self,dispersed):
        print("H segments")
        a,b = self.H_PCA_sens.shape
        count= int(b/100)
        d_all = np.zeros(b)
        for i in range(count):
            H = self.H_PCA_sens[:,i*100:(i+1)*100]
            f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
            result = lsqr(H,f, iter_lim=100, show=False) #solves min_d ||Ad-f||^2.
            d = result[0] #  lsqr stores result as final_solution, istop, itn.... So we use only [0]
            d_all[i*100:(i+1)*100] = d
            print(count-i)
        A = build_matrix()
        Recovered = A.integrated_flux_image_PCA(d_all)
     
        return Recovered
        
        
