from grismagic.traces import GrismTrace
from build_matrix import build_matrix
from dispersion import dispersion
from recovery import recovery
import numpy as np

import matplotlib.pyplot as plt
import time 

from pathlib import Path
import pandas as pd

# Globally define x range, y range and basis length that are used to construct matrix
x_pixel = 20
y_pixel= 500
basis_length=10

# load paths of config files for matrix construction
base = Path(__file__).parent

config = base / "Config Files" / "GR150R.F150W.220725.conf"
wave = base / "jwst_niriss_wavelengthrange_0002.asdf"

# Constructs matrix. Do this ONCE
#H = build_matrix(config, filter_name="F150W", wavelengthrange_file=wave, rows=y_pixel, columns=x_pixel)
#H.build_and_save_trace_matrix_coefficients_PCA_sensitivity()

######################## direct image ############################
# build mock stars
def random_stars_PCA():
    """Constructs vector a_tilde where every basis_length=10 entries belong to one pixel. Most of them are zero and for p pixels assign random
    values a_i(x,y), such that a_i(x,y)*phi_i(lambda) is positive for all lambdas, to ensure positive flux, i.e. physical meaning"""
    
    A = build_matrix(config, filter_name="F150W", wavelengthrange_file=wave, rows=y_pixel, columns=x_pixel) # calls build_matrix since basis method is there
    basis = A.eigenspectra_basis() # constructs basis
        
    p=0.1 # p pixels of the image have sources
    
    N = x_pixel*y_pixel # image size 

    a_tilde = np.zeros(N*basis_length) # constructs a_tilde with correct size. a_tilde will be 0 everywhere without source
    
    num_active = int(p*N) # p*100 percent will have sources assigned
    active_k= np.random.choice(N, size= num_active, replace= False) # randomly chooses sources
    
    max_tries = 50 # maximum tries to find a positive flux coefficient set
    
    for k in active_k: # for all prior chosen source pixels

        for _ in range(max_tries):
            flux = np.random.uniform(-1, 1, size=basis_length) # randomly chooses coefficients 
            spectrum = basis @ flux  # computes f(x,y,lambda)= sum_i a_i(x,y)phi_i(lambda)
            
            if np.all(spectrum >= 0): # only allows positive flux
                break
        else:
            # fallback if no valid sample found
            flux = np.zeros(basis_length) # so possibly less than p sources in the end
            
        # assign to correct block in flattened vector
        a_tilde[k * basis_length : (k + 1) * basis_length] = flux
    return a_tilde


# constructs coefficient vector a_tilde
start = time.time()
a_tilde = random_stars_PCA()

# a_tilde is just a coefficient vector, build.integrated_flux_image_PCA(a_tilde) makes it visible as image
build = build_matrix(config, filter_name="F150W", wavelengthrange_file=wave)
mock_direct = build.integrated_flux_image_PCA(a_tilde) # make direct image visible

# generate some noise
#noise = np.random.uniform(0, 1000, size=mock_direct.shape)
#mock_direct = mock_direct + noise

np.save(f"mock_{x_pixel}_{y_pixel}.npy",mock_direct) # saves direct image
end = time.time()
disptime = end - start
print( f"Dirrect image construction (includs saving):{disptime:.3f} s")

############################# dispersion ##################################################

start = time.time()
disp = dispersion()
mock_dispersed = disp.dispersed_PCA(a_tilde) # compute dispersed image
np.save("mock_dispersed_{x_pixel}_{y_pixel}.npy", mock_dispersed)
end = time.time()
disptime = end - start
print( f"Dispersion (includs saving):{disptime:.3f} s")

# initial guess
#   takes direct image as image, so not the coefficients, and sets all coeff to zero except the 10 coeff
#   that correspond to its pixel, they are set to 1. In recovery function, this is trimming the matrix H to just the active sources

coords =np.where(mock_direct!=0) # in mock the possible stars have currently value greater than zero
#coords =np.where(mock_direct>np.mean(mock_direct)/10000) # cutting out noise. in mock the possible stars have currently value greater than zero
# print(np.mean(mock_direct)/10000)
possible_stars = list(zip(coords[0], coords[1])) # converte to list s.t. possible_stars[i]=(y_i,x_i)
coefs = np.zeros(y_pixel*x_pixel*10)

for i in range(len(possible_stars)): # initial guess = 1 for all
    y,x=possible_stars[i]
    pixel = y*x_pixel+x
    coefs[pixel*basis_length : (pixel+1)*basis_length] = 1
    
###################### recovery ##########################################
start = time.time()
recov = recovery()
d= recov.recover_direct_from_traces_basis_matrix_PCA(mock_dispersed, image=False, initial_guess=coefs) # recovers image. image=False to output the vector d and not the ready image
end = time.time()
disptime = end - start
print( f"Recovery :{disptime:.3f} s")
###################### visualization of spectra #######################
Phi = build.eigenspectra_basis()
all_true_vals = []
all_rec_vals  = []


for i in range(int(len(a_tilde)/10)):
    if np.any(a_tilde[i*10:(i+1)*10]!= 0):
        
        spectrum = Phi @ d[i*10:(i+1)*10]#recovered spectrum
        spectrum_og = Phi @ a_tilde[i*10:(i+1)*10] # original spectrum
        
###################### plot both spectra against each other ##################
        # plt.subplot(1,2,1)
        # plt.plot(build.lambdas, spectrum_og)
        # plt.xlabel("Wavelength in Ångström")
        # plt.ylabel("Flux")
        # plt.title(f"Original spectrum at k=y*20+x= {i}")
        # plt.subplot(1,2,2)
        # plt.plot(build.lambdas, spectrum)
        # plt.xlabel("Wavelength in Ångström")
        # plt.ylabel("Flux")
        # plt.title(f"Recovered spectrum at k=y*20+x= {i}")
        # plt.show()

        # accumulate instead of plotting
        all_true_vals.append(spectrum_og)
        all_rec_vals.append(spectrum)

######################### single global parity plot #############################
# stack everything into one big vector
true_vals = np.concatenate(all_true_vals)
rec_vals  = np.concatenate(all_rec_vals)


plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
#plt.hexbin(true_vals, rec_vals, gridsize=300, mincnt=1, cmap='viridis')
plt.scatter(true_vals, rec_vals, s=2, alpha =0.01)


plt.gca().set_aspect('equal', adjustable='box')

minv= min(true_vals.min(), rec_vals.min())
maxv = max(true_vals.max(), rec_vals.max())
plt.plot([minv, maxv], [minv, maxv], 'r--')
plt.xlim(minv, maxv)
plt.ylim(minv, maxv)
plt.xlabel("original f(λ)")
plt.ylabel("reconstructed f(λ)")
plt.title("Parity plot (all points)")

plt.show()
        
################################# saves recovery #######################################


mock_recovered = build.integrated_flux_image_PCA(d) # converts recovered to visible image
np.save("mock_recovered_{x_pixel}_{y_pixel}.npy", mock_recovered)

########################## visualize images #######################################
# loads files
base = Path(__file__).resolve().parent

mock_direct = np.load(base / "mock_20_500.npy")
mock_dispersed = np.load(base / "mock_dispersed_20_500.npy")
mock_recovered = np.load(base / "mock_recovered_20_500.npy")

#################################################################
# plots files as images
plt.subplot(1,4,1)
std1 = np.nanstd(mock_direct)
mean1 = np.nanmean(mock_direct)
plt.imshow(mock_direct, cmap="inferno", vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Direct")

plt.subplot(1,4,2)
std2 = np.nanstd(mock_dispersed)
mean2 = np.nanmean(mock_dispersed)
plt.imshow(mock_dispersed, cmap="inferno", vmin=-(mean2 + 2*std2), vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Dispersed")

# plt.subplot(1,6,3)
# H= load_npz("H_matrix_F150W_flux_20_500_orders_PCA_sensitivity.npz")
# mock_dispersed_residual = (H@d).reshape(500,20)
# std2 = np.nanstd(mock_dispersed_residual)
# mean2 = np.nanmean(mock_dispersed_residual)
# plt.imshow(mock_dispersed_residual, cmap="inferno", vmin=-(mean2 + 2*std2), vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
# plt.colorbar()
# plt.title("Dispersed of LSQR H*d")

plt.subplot(1,4,3)
plt.imshow(mock_recovered, vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, cmap="inferno", interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Recovered")

plt.subplot(1,4,4)
#mock_direct[mock_direct==0]=1e-14
std1 = np.nanstd(np.abs(mock_direct-mock_recovered))
mean1 = np.nanmean(np.abs(mock_direct-mock_recovered))
plt.imshow(np.abs(mock_direct-mock_recovered), vmin=0, vmax=mean1 + 1*std1, cmap="inferno", interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Rediduals: |Direct - Recovered|")
plt.show()

# plt.subplot(1,6,6)

# std2 = np.nanstd(mock_dispersed-mock_dispersed_residual)
# mean2 = np.nanmean(mock_dispersed-mock_dispersed_residual)
# plt.imshow(mock_dispersed-mock_dispersed_residual, cmap="inferno", vmin=-(mean2 + 2*std2), vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
# plt.colorbar()
# plt.title("mock_dispersed-mock_dispersed_residual")
# plt.show()
