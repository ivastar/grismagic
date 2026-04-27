from grismagic.traces import GrismTrace
from build_matrix import build_matrix
from dispersion import dispersion
from recovery import recovery
import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import time 
import asdf
from pathlib import Path
from astropy.stats import sigma_clip
import pandas as pd
from scipy.linalg import pinv
from scipy.sparse import load_npz, save_npz
from scipy.ndimage import binary_dilation, label
from scipy.ndimage import generate_binary_structure

x_pixel = 20
y_pixel= 500
basis_length=10

def random_stars_PCA():
    
    A = build_matrix()
    basis = A.eigenspectra_basis()
        
    p=0.1
    N = x_pixel*y_pixel
    n = 10
    a_tilde = np.zeros(N*n)
    
    num_active = int(p*N)
    active_k= np.random.choice(N, size= num_active, replace= False)
    
    max_tries = 50
    for k in active_k:

        for _ in range(max_tries):
            flux = np.random.uniform(-1, 1, size=n)
            spectrum = basis @ flux  # shape (150,)
            
            if np.all(spectrum >= 0):
                break
        else:
            # fallback if no valid sample found
            flux = np.zeros(n)
        # assign to correct block in flattened vector
        a_tilde[k * n : (k + 1) * n] = flux
    return a_tilde
H = build_matrix("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F150W.220725.conf",filter_name="F150W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")
H.build_and_save_trace_matrix_coefficients_PCA_sensitivity()

######################
##########################
##########################
# mock image. star consisting of 5 pixels centered at x,y= 50,490

# a_star = np.array([1.2,-0.4,0.08,0.02,-0.01,0.005,0.1,0,0,0]) #manual spectral coefficients
# a_star1 = np.array([1.0,-0.4,0.08,0.02,0.05,-0.008,0,0.2,0,0]) #manual spectral coefficients
# a_star2 = np.array([1.2,0.4,0.02,-0.02,0.01,0.01,0.4,0,0,0]) #manual spectral coefficients
# a_star3 = np.array([2,-0.4,0.08,0.02,-0.01,0.005,0,0,0,2]) #manual spectral coefficients
# a_star4 = np.array([-0.1,-0.6,0.08,0.02,-0.01,0.005,0,0,1,0]) #manual spectral coefficients

# x=10
# y=250
# a_tilde = np.zeros(500*20*10)
# pixel_star = y*20+x
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star    # all 5 pixels belong to the same star so same spectral coefficients

# x=17
# y=200
# pixel_star = y*20+x
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star1

# x=6
# y=50
# pixel_star = y*20+x
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star2

# x=10
# y=170
# pixel_star = y*20+x
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star3

# x=19
# y=300
# pixel_star = y*20+x
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star4
#####################################
####################################
a_tilde = random_stars_PCA()


build = build_matrix()
mock_direct = build.integrated_flux_image_PCA(a_tilde) # make direct image visible

# generate some noise
#noise = np.random.uniform(0, 1000, size=mock_direct.shape)
#mock_direct = mock_direct + noise

np.save("mock_20_500.npy",mock_direct)

disp = dispersion()
mock_dispersed = disp.dispersed_PCA(a_tilde) # compute dispersed image
np.save("mock_dispersed_20_500.npy", mock_dispersed)

######################################
# initial guess
#   takes direct image as image, so not the coefficients, and sets all coeff to zero except the 10 coeff
#   that correspond to its pixe, they are set to 1.

coords =np.where(mock_direct!=0) # in mock the possible stars have currently value greater than zero
#coords =np.where(mock_direct>np.mean(mock_direct)/10000) # cutting out noise. in mock the possible stars have currently value greater than zero
print(np.mean(mock_direct)/10000)
possible_stars = list(zip(coords[0], coords[1])) # converte to list s.t. possible_stars[i]=(y_i,x_i)
coefs = np.zeros(y_pixel*x_pixel*10)

for i in range(len(possible_stars)):
    y,x=possible_stars[i]
    pixel = y*x_pixel+x
    coefs[pixel*basis_length : (pixel+1)*basis_length] = 1
######################################
recov = recovery()
d= recov.recover_direct_from_traces_basis_matrix_PCA(mock_dispersed, image=False, initial_guess=coefs) # recovers image. image=False to output the vector d and not the ready image


Phi = build.eigenspectra_basis()
k = np.nonzero(a_tilde)[0][0]

n = 10

spectrum = Phi @ d[k:k+n]#recovered spectrum
spectrum_og = Phi @ a_tilde[k:k+n] # original spectrum
###################### plot both spectra against each other
plt.subplot(1,2,1)
plt.plot(build.lambdas, spectrum_og)
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title(f"Original spectrum k=y*20+x= {k}")
plt.subplot(1,2,2)
plt.plot(build.lambdas, spectrum)
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title(f"Recovered spectrum k=y*20+x= {k}")
plt.show()
########################## extracts spectrum of pixel
# Phi = build.eigenspectra_basis()

# x = 10
# y= 250
# k = y*20+x 
# n = 10
# a_k = d[k*n:(k+1)*n]
# spectrum = Phi @ a_k #recovered spectrum
# spectrum_og = Phi @ a_star # original spectrum
# ###################### plot both spectra against each other
# plt.subplot(1,2,1)
# plt.plot(build.lambdas, spectrum_og)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Original spectrum (x,y)= ({x},{y})")

# plt.subplot(1,2,2)
# plt.plot(build.lambdas, spectrum)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Recovered spectrum (x,y)= ({x},{y})")
# plt.show()
# #######################
# x=17
# y=200
# k = y*20+x 
# n = 10
# a_k = d[k*n:(k+1)*n]
# spectrum = Phi @ a_k #recovered spectrum
# spectrum_og = Phi @ a_star1 # original spectrum
# ###################### plot both spectra against each other
# plt.subplot(1,2,1)
# plt.plot(build.lambdas, spectrum_og)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Original spectrum (x,y)= ({x},{y})")

# plt.subplot(1,2,2)
# plt.plot(build.lambdas, spectrum)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Recovered spectrum (x,y)= ({x},{y})")
# plt.show()
# ############################################
# x=6
# y=50
# k = y*20+x 
# n = 10
# a_k = d[k*n:(k+1)*n]
# spectrum = Phi @ a_k #recovered spectrum
# spectrum_og = Phi @ a_star2 # original spectrum
# ###################### plot both spectra against each other
# plt.subplot(1,2,1)
# plt.plot(build.lambdas, spectrum_og)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Original spectrum (x,y)= ({x},{y})")

# plt.subplot(1,2,2)
# plt.plot(build.lambdas, spectrum)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Recovered spectrum (x,y)= ({x},{y})")
# plt.show()
# ################################################
# x=10
# y=170
# k = y*20+x 
# n = 10
# a_k = d[k*n:(k+1)*n]
# spectrum = Phi @ a_k #recovered spectrum
# spectrum_og = Phi @ a_star3 # original spectrum
# ###################### plot both spectra against each other
# plt.subplot(1,2,1)
# plt.plot(build.lambdas, spectrum_og)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Original spectrum (x,y)= ({x},{y})")

# plt.subplot(1,2,2)
# plt.plot(build.lambdas, spectrum)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Recovered spectrum (x,y)= ({x},{y})")
# plt.show()
# #############################
# x=19
# y=300
# k = y*20+x 
# n = 10
# a_k = d[k*n:(k+1)*n]
# spectrum = Phi @ a_k #recovered spectrum
# spectrum_og = Phi @ a_star4 # original spectrum
# ###################### plot both spectra against each other
# plt.subplot(1,2,1)
# plt.plot(build.lambdas, spectrum_og)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Original spectrum (x,y)= ({x},{y})")

# plt.subplot(1,2,2)
# plt.plot(build.lambdas, spectrum)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title(f"Recovered spectrum (x,y)= ({x},{y})")
# plt.show()
#############################################################


mock_recovered = build.integrated_flux_image_PCA(d) # converts recovered to visible image
np.save("mock_recovered_20_500.npy", mock_recovered)

# #################################################################
# base = Path(__file__).resolve().parent

# mock_direct = np.load(base / "mock_20_500.npy")
# mock_dispersed = np.load(base / "mock_dispersed_20_500.npy")
# mock_recovered = np.load(base / "mock_recovered_20_500.npy")
# #################################################################
# plt.subplot(1,6,1)
# std1 = np.nanstd(mock_direct)
# mean1 = np.nanmean(mock_direct)
# plt.imshow(mock_direct, cmap="inferno", vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, interpolation="nearest", origin="lower",aspect="auto")
# plt.colorbar()
# plt.title("Mock direct")

# plt.subplot(1,6,2)
# std2 = np.nanstd(mock_dispersed)
# mean2 = np.nanmean(mock_dispersed)
# plt.imshow(mock_dispersed, cmap="inferno", vmin=-(mean2 + 2*std2), vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
# plt.colorbar()
# plt.title("Mock dispersed")

# plt.subplot(1,6,3)
# H= load_npz("H_matrix_F150W_flux_20_500_orders_PCA_sensitivity.npz")
# mock_dispersed_residual = (H@d).reshape(500,20)
# std2 = np.nanstd(mock_dispersed_residual)
# mean2 = np.nanmean(mock_dispersed_residual)
# plt.imshow(mock_dispersed_residual, cmap="inferno", vmin=-(mean2 + 2*std2), vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
# plt.colorbar()
# plt.title("Dispersed of LSQR A*d")

# plt.subplot(1,6,4)
# plt.imshow(mock_recovered, vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, cmap="inferno", interpolation="nearest", origin="lower",aspect="auto")
# plt.colorbar()
# plt.title("Mock recovered")

# plt.subplot(1,6,5)
# mock_direct[mock_direct==0]=1e-14
# std1 = np.nanstd(np.abs(mock_direct-mock_recovered)/mock_direct)
# mean1 = np.nanmean(np.abs(mock_direct-mock_recovered)/mock_direct)
# plt.imshow(np.abs(mock_direct-mock_recovered)/mock_direct, vmin=0, vmax=mean1 + 1*std1, cmap="inferno", interpolation="nearest", origin="lower",aspect="auto")
# plt.colorbar()
# plt.title("Rediduals: |(mock_direct-mock_recovered)|/mock_direct")


# plt.subplot(1,6,6)

# std2 = np.nanstd(mock_dispersed-mock_dispersed_residual)
# mean2 = np.nanmean(mock_dispersed-mock_dispersed_residual)
# plt.imshow(mock_dispersed-mock_dispersed_residual, cmap="inferno", vmin=-(mean2 + 2*std2), vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
# plt.colorbar()
# plt.title("mock_dispersed-mock_dispersed_residual")
# plt.show()
