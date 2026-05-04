import numpy as np 
from grismagic.traces import GrismTrace
from astropy.io import fits
from scipy.interpolate import interp1d
import pandas as pd
from scipy.sparse import save_npz, lil_matrix
from pathlib import Path

class build_matrix: 
    def __init__(self, config, filter_name,wavelengthrange_file, rows=500, columns=20):
        """Important: The wavelength has to be in Angstrom since the coefficients are calibrated to angstrom.
        Initialized by Config file (XRANGE, YRANGE, DFILTER, DYDX_A_0,DYDY_A_1,DLDP_A_0,DLDP_A_1)
        """
        
        self.tr = GrismTrace.from_file(config, filter_name, wavelengthrange_file)  # auto-detects configurations

        self.lo, self.hi = self.tr._lam_range("1", None, None) #minimum and maximum wavelength in microns for first order. All orders have same range
        self.lo = int(self.lo * 10000) #as int in Angstrom
        self.lo = max(self.lo, 7000) #for PCA, goes from 0.7 to 2.2
        self.hi= int(self.hi * 10000)   # as int in Angstrom
        self.hi = min(self.hi, 22000) #for PCA, goes from 0.7 to 2.2
        
        self.lambdas = np.linspace(self.lo, self.hi, 150) #wavelength list
  
        self.orders = self.tr.orders
        
        self.xmin, self.xmax = 0, rows # pixel row range 0,500
        self.ymin, self.ymax = 0, columns # pixel column range 0,20
        self.x_pixel = np.abs(self.xmin) + np.abs(self.xmax) #we need the size of the image in coding coordinates
        self.y_pixel = np.abs(self.ymin) + np.abs(self.ymax) 
        
        self.df = pd.read_csv("eigenspectra_kurucz.csv", sep=",") #extracts eigenspectra from pca file 
        


#######################################################
# PCA eigenspectra basis
#######################################################
    def eigenspectra_basis(self):
        """Basis by using eigenspectra from PCA. Wavelengths are in um!!!"""
        wavelength = self.df.iloc[:, 0].to_numpy() #the wavelengths from pca (lambdas, 1)
        eigenspectra = self.df.iloc[:, 1:].to_numpy() #the 10 eigenspectra in (lambdas, 10)

        wavelength_angstrom = wavelength* 1e4 # conversion from um to angstrom
        interp_func = interp1d(wavelength_angstrom, eigenspectra, axis=0, kind='linear') #interpolation to use self.lambdas

        eigenspectra_new = interp_func(self.lambdas)  # shape (self.lambdas, 10)
        
        return eigenspectra_new


#####################################
# Matrix with basis for all orders with PCA basis and sensitivity curves
####################################


    def build_trace_matrix_coefficients_PCA_sensitivity(self):
        """Builds the matrix H with size(x_pixel*y_pixel)*(x_pixel*y_pixel*h) where each row is a pixel in the dispersed image and each column 
        the trace at a basis function at the object in direct coordinates. Uses the spectrum function phi_m(lambda), s.t. we store phi in H and only need a_m(x,y) in d. 
        Called by def build_and_save_matrix
            f = H @ a_tilde
            where a_tilde consists out of a_0(k_1),a_1(k_1)....a_n(k_last)
        """
        ########
        # sensitivity curves
        base = Path(__file__).parent

        SenseConfig = base / "SenseConfig" / "wfss-grism-configuration" / "NIRISS.GR150R.F150W.1.etc.1.5.2.sens.fits"
        hdu = fits.open(SenseConfig) #F150W, GR150R
        data1= hdu[1].data
        wavelength1 = data1["WAVELENGTH"]
        sensitivity1 = data1["SENSITIVITY"]
        max1 = np.sum(sensitivity1)
        sensitivity1=sensitivity1/max1 #normalized

        hdu.close()
        SenseConfig = base / "SenseConfig" / "wfss-grism-configuration" / "NIRISS.GR150R.F150W.0.etc.1.5.2.sens.fits"
        hdu = fits.open(SenseConfig) #F150W, GR150R
        data0= hdu[1].data
        wavelength0 = data0["WAVELENGTH"]
        sensitivity0 = data0["SENSITIVITY"] 
        max0= np.sum(sensitivity0)
        sensitivity0 = sensitivity0/max0 #normalized by the same factor as 1st order

        hdu.close()
        SenseConfig = base / "SenseConfig" / "wfss-grism-configuration" / "NIRISS.GR150R.F150W.2.etc.1.5.2.sens.fits"
        hdu = fits.open(SenseConfig) #F150W, GR150R
        data2= hdu[1].data
        wavelength2 = data2["WAVELENGTH"]
        sensitivity2 = data2["SENSITIVITY"] 
        max2 = np.sum(sensitivity2)
        sensitivity2 = sensitivity2/max2 #normalized by the same factor as 1st order

        hdu.close()
        sens_interp = [interp1d(wavelength1, sensitivity1, bounds_error=False, fill_value=0.0),interp1d(wavelength0, sensitivity0, bounds_error=False, fill_value=0.0),interp1d(wavelength2, sensitivity2, bounds_error=False, fill_value=0.0)]


        #assembles matrix H
        N = self.x_pixel * self.y_pixel # row dimension of H
        p,q = self.df.shape
        h=q-1 # amount of basis functions. -1 because 1st column are the wavelengths.
   
        H=lil_matrix((N,N*h)) #good for sparse matrices
        Phi = self.eigenspectra_basis()
      
        
        order102 = ["A","B","C"]
        for order in order102: # Different notation: A=1, B=0, C=2 ....
            if order == "A":
                senscount = 1
                Phi = Phi * sens_interp[senscount](self.lambdas)[:,None] # rescales basis functions with sensitivity curves
            elif order == "B":
                senscount = 0
                Phi = Phi * sens_interp[senscount](self.lambdas)[:,None] # rescales basis functions with sensitivity curves
            elif order == "C":
                senscount = 2
                Phi = Phi * sens_interp[senscount](self.lambdas)[:,None] # rescales basis functions with sensitivity curves
            print(order)
            for i in range(self.x_pixel):
                for j in range(self.y_pixel):
                    k= i*self.y_pixel +j #gets column index right, for which pixel/column are we inserting the trace
                    
                    x0 = i + self.xmin
                    y0 = j + self.ymin

                    try:
                        # CRUCIAL: align wavelengths with your basis
                        x_trace, y_trace = self.tr.get_trace_at_wavelength(x0, y0, order=order, lam= self.lambdas)
                    except:
                        continue
                    
                    x_trace = np.array(x_trace)
                    y_trace = np.array(y_trace)
                    

                    # pixel mapping
                    x_pix = np.round(x_trace).astype(int)
                    y_pix = np.round(y_trace).astype(int)

                    mask = (
                        (x_pix >= self.xmin) & (x_pix < self.xmax) &
                        (y_pix >= self.ymin) & (y_pix < self.ymax)
                    )
                    if not np.any(mask):
                        continue


                    x_valid = x_pix[mask] # only x values that are visible in x range
                    y_valid = y_pix[mask] # only y values that are visible in y range
                    lam_idx = np.where(mask)[0]
                    
                    rows = (x_valid - self.xmin)*self.y_pixel +(y_valid-self.ymin) # gets row index right for trace. In which rows appears the trace?

   
                    for idx, row in enumerate(rows): # enumerates gives back the index of the row and the row at the same time.
                        l_indx = lam_idx[idx]

                        for m in range(h):
                            col = k*h+ m #correct column indexing
                            H[row,col] += Phi[l_indx,m] # instead of a trace 00111110000 we add now the function phi(lambdas) to positions of the trace
                
        return H
    
    def build_and_save_trace_matrix_coefficients_PCA_sensitivity(self):
        """Calls build and saves the template matrix containing all traces. 
        Furthermore A stores in its last row the amount of ones per column to determine how many
        colored pixels each trace has. JUST DO THIS ONCE PER CONFIGURATION"""
        
        H = self.build_trace_matrix_coefficients_PCA_sensitivity()
        H = H.tocsr()
        save_npz(f"H_matrix_F150W_flux_{self.ymax}_{self.xmax}_orders_PCA_sensitivity.npz", H)
        return

##################################################
# PCA two functions to construct image from coefficients vector
########################################################
    def compute_phi_weights_PCA(self):
        """Approximates I(x,y)=int a(x,y)phi(lambda) d lambda"""
        Phi = self.eigenspectra_basis()  # shape (n_lambda, 10)
        delta = self.lambdas[1]-self.lambdas[0]
    
        # integrate each basis function over lambda
        w = np.sum(Phi, axis=0)*delta   # shape (n_lambda,)
        return w
    
    def integrated_flux_image_PCA(self, a_tilde):
        """creates direct image from coefficients with I(x,y)=int a(x,y)phi(lambda) d lambda"""
        Phi = self.eigenspectra_basis()  # shape (n_lambda, 10)
        n = Phi.shape[1] # =10
        w = self.compute_phi_weights_PCA()
    
        image = np.zeros((self.x_pixel, self.y_pixel))
    
        for k in range(self.x_pixel * self.y_pixel):
            i = k // self.y_pixel
            j = k % self.y_pixel
        
            a_k = a_tilde[k*n:(k+1)*n]
        
            image[i,j] = np.dot(a_k, w)
    
        return image
    