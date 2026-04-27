import numpy as np 
from grismagic.traces import GrismTrace
from astropy.io import fits
from scipy.interpolate import interp1d
import pandas as pd
from scipy.sparse import save_npz, lil_matrix, vstack
from pathlib import Path

class build_matrix: 
    def __init__(self, config, filter_name,wavelengthrange_file):
        """Important: The wavelength has to be in Angstrom since the coefficients are calibrated to angstrom.
        Initialized by Config file (XRANGE, YRANGE, DFILTER, DYDX_A_0,DYDY_A_1,DLDP_A_0,DLDP_A_1)
        """
        
        self.tr = GrismTrace.from_file(config, filter_name, wavelengthrange_file)  # auto-detects format

        self.lo, self.hi = self.tr._lam_range("1", None, None) #minimum and maximum wavelength in microns for first
        self.lo = int(self.lo * 10000) #as int
        self.lo = max(self.lo, 7000) #for PCA, goes from 0.7 to 2.2
        self.hi= int(self.hi * 10000)   # as int
        self.hi = min(self.hi, 22000) #for PCA, goes from 0.7 to 2.2
        self.lambdas = np.linspace(self.lo, self.hi, 150) #wavelength list
        self.base = np.linspace(self.lo, self.hi, 5) #wavelength list
        self.mu = np.linspace(self.lambdas.min(), self.lambdas.max(), len(self.base))
        delta = self.mu[1]-self.mu[0]
        self.sigma = 0.7*delta*np.ones_like(self.mu)
        self.orders = self.tr.orders
        
        self.xmin, self.xmax = 0, 2048 # pixel range
        self.ymin, self.ymax = 0, 2048
        self.x_pixel = np.abs(self.xmin) + np.abs(self.xmax) #we need the size of the image in coding coordinates
        self.y_pixel = np.abs(self.ymin) + np.abs(self.ymax) 
        
        self.df = pd.read_csv("eigenspectra_kurucz.csv", sep=",") #extracts eigenspectra from pca file 
        


####################################
# hat function basis
#####################################
            
    def hat_function_phi_m(self,lam,m):
        """definition of basis hat functions phi for flux"""

        delta = self.base[1]-self.base[0]
        return np.maximum(0,(1-np.abs(lam-self.base[m])/delta)) # 1 elsewise to create continuos trace
    
    def spectral_basis(self):
        """Basis out of all functions phi_m, where m=1 to n and n is the number of wavelegths that we observe"""
        n = len(self.lambdas)

        Phi = np.zeros((n,len(self.base)))
        
        for m in range(len(self.base)):
            Phi[:,m] = self.hat_function_phi_m(self.lambdas,m) #builds matrix where each row is for a given l_t -> [phi_0(l_t), phi_(l_t),...,phi_n(l_t)]
        
        return Phi
    
    
    #########################################
    # gaussian basis
    #############################################
    def gaussian_basis_function_phi_m(self, lam, m):
        mu_m = self.mu[m]
        sigma_m = self.sigma[m]
        return np.exp(-(lam - mu_m)**2 / (2 * sigma_m**2))
    
    def spectral_basis_gaussian(self):
        n_lambda = len(self.lambdas)
        n_basis = len(self.mu)

        Phi = np.zeros((n_lambda, n_basis))

        for m in range(n_basis):
            Phi[:, m] = self.gaussian_basis_function_phi_m(self.lambdas, m)

        return Phi
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

##################################################
# two functions to construct image from coefficients vector
########################################################
    def compute_phi_weights(self):
        """Approximates I(x,y)=int a(x,y)phi(lambda) d lambda"""
        Phi = self.spectral_basis_gaussian()  # shape (n_lambda, n_lambda)
        delta = self.base[1] - self.base[0]
    
        # integrate each basis function over lambda
        w = np.sum(Phi, axis=0)*delta   # shape (n_lambda,)
        return w
    
    def integrated_flux_image(self, a_tilde):
        """creates direct image from coefficients with I(x,y)=int a(x,y)phi(lambda) d lambda"""
        n = len(self.base)
        w = self.compute_phi_weights()
    
        image = np.zeros((self.x_pixel, self.y_pixel))
    
        for k in range(self.x_pixel * self.y_pixel):
            i = k // self.y_pixel
            j = k % self.y_pixel
        
            a_k = a_tilde[k*n:(k+1)*n]
        
            image[i,j] = np.dot(a_k, w)
    
        return image
    
    
    #####################################
    # Matrix with basis for one order
    ####################################


    def build_trace_matrix_coefficients(self):
        """Builds the matrix H with size(x_pixel*y_pixel)*(x_pixel*y_pixel*h) where each row is a pixel in the dispersed image and each column 
        the trace at a basis function at the object in direct coordinates. Uses the spectrum function phi_m(lambda), s.t. we store phi in H and only need a_m(x,y) in d. 
        Called by def build_and_save_matrix
            f = H @ a_tilde
            where a_tilde consists out of a_0(k_1),a_1(k_1)....a_n(k_last)
        """
        #assembles matrix H
        N = self.x_pixel * self.y_pixel # row dimension of H
        h = len(self.base)
        H=lil_matrix((N,N*h)) #good for sparse matrices
        Phi = self.spectral_basis_gaussian()

        for i in range(self.x_pixel):
            for j in range(self.x_pixel):
                k= i*self.y_pixel +j #gets column index right, for which pixel/column are we inserting the trace
                
                x0 = i + self.xmin
                y0 = j + self.ymin

                try:
                    # CRUCIAL: align wavelengths with your basis
                    x_trace, y_trace = self.tr.get_trace_at_wavelength(x0, y0, order="A", lam=self.lambdas)
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
                lam_indices = np.where(mask)[0]
                
                rows = (x_valid - self.xmin)*self.y_pixel +(y_valid-self.ymin) # gets row index right for trace. In which rows appears the trace?
        
            
                for idx, row in enumerate(rows): # enumerates gives back the index of the row and the row at the same time.
                    l_indx = lam_indices[idx]
                    for m in range(h):
                        col = k*h+ m #correct column indexing
                        H[row,col] += Phi[l_indx,m] # instead of a trace 00111110000 we add now the function phi(lambdas) to positions of the trace
            
        return H
    
    def build_and_save_trace_matrix_coefficients(self):
        """Calls build and saves the template matrix containing all traces. 
        Furthermore A stores in its last row the amount of ones per column to determine how many
        colored pixels each trace has. JUST DO THIS ONCE PER CONFIGURATION"""
        
        H = self.build_trace_matrix_coefficients()
        H = H.tocsr()
        save_npz("H_matrix_flux.npz", H)
        return
    
    
    ##################################################
    # ones and zeros trace matrix with trace count
    ###################################################
    
    def build_trace_matrix(self):
        """Builds the matrix A with size(x_pixel*y_pixel)*(x_pixel*y_pixel+1) where each row is a pixel in the dispersed image and each column 
        the trace at the object in direct coordinates. Last row includes how many trace pixels per column. Called by def build_and_save_matrix"""

        trace_cache ={}#precompute traces
        for i in range (self.x_pixel): #iterating over all pixels
                for j in range (self.y_pixel): 
                                # grismagic expects source positions in detector coords
                    x0 = self.xmin + i
                    y0 = self.ymin + j
                    # Compute the trace offsets along the grism
                    # offset=None lets grismagic choose the full trace range
                    x_trace, y_trace,_ = self.tr.get_trace(x0, y0, order="A")
                
                    x_new = np.round(x_trace).astype(int) #trace in observation coordinates
                    y_new = np.round(y_trace).astype(int) 
                
                    #only visible trace
                    mask = (x_new >= self.xmin) & (x_new < self.xmax) & (y_new >= self.ymin) & (y_new < self.ymax) 
                    
                    
                    trace_cache[(i,j)]=(x_new, y_new, mask)
                    
        #assembles matrix A
        N = self.x_pixel * self.y_pixel
        A=lil_matrix((N,N)) #good for sparse matrices
        for (i,j),(x_new,y_new,mask) in trace_cache.items():
            k= i*self.y_pixel +j
            
            mask_indices = np.where(mask)[0]
            if len(mask_indices)==0:
                continue

            x_valid = x_new[mask] 
            y_valid = y_new[mask] 
            
            rows = (x_valid - self.xmin)*self.y_pixel +(y_valid-self.ymin)
            A[rows,k] = 1
            
        # compute row sum to mark how many trace pixels per column
        row_sum = A.sum(axis = 0)
        
        # Append it as last row to A
        A = vstack([A, row_sum])
        return A
    
    def build_and_save_trace_matrix(self):
        """Calls build and saves the template matrix containing all traces. 
        Furthermore A stores in its last row the amount of ones per column to determine how many
        colored pixels each trace has. JUST DO THIS ONCE PER CONFIGURATION"""
        A = self.build_trace_matrix()
        A = A.tocsr()
        save_npz("A_matrix_with_trace_count.npz", A)
        return
    
    
    ##########################################
    # All orders and basis functions
    ###########################################
    
    def build_trace_matrix_coefficients_orders(self):
        """Builds the matrix H with size(x_pixel*y_pixel)*(x_pixel*y_pixel*h) where each row is a pixel in the dispersed image and each column 
        the trace at a basis function at the object in direct coordinates. Uses the spectrum function phi_m(lambda), s.t. we store phi in H and only need a_m(x,y) in d. 
        Called by def build_and_save_matrix
            f = H @ a_tilde
            where a_tilde consists out of a_0(k_1),a_1(k_1)....a_n(k_last)
        loops over all available orders
        """
        #assembles matrix H
        N = self.x_pixel * self.y_pixel # row dimension of H
        h = len(self.base)
        H=lil_matrix((N,N*h)) #good for sparse matrices
        Phi = self.spectral_basis_gaussian()

        for order in self.orders:
            for i in range(self.x_pixel):
                for j in range(self.x_pixel):
                    k= i*self.y_pixel +j #gets column index right, for which pixel/column are we inserting the trace
                    
                    x0 = i + self.xmin
                    y0 = j + self.ymin

                    try:
                        # align wavelengths with your basis
                        x_trace, y_trace = self.tr.get_trace_at_wavelength(x0, y0, order=order, lam=self.lambdas)
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
                    lam_indices = np.where(mask)[0]
                    
                    rows = (x_valid - self.xmin)*self.y_pixel +(y_valid-self.ymin) # gets row index right for trace. In which rows appears the trace?
            
                
                    for idx, row in enumerate(rows): # enumerates gives back the index of the row and the row at the same time.
                        l_indx = lam_indices[idx]
                        for m in range(h):
                            col = k*h+ m #correct column indexing
                            H[row,col] += Phi[l_indx,m] # instead of a trace 00111110000 we add now the function phi(lambdas) to positions of the trace
                
        return H
    
    def build_and_save_trace_matrix_coefficients_orders(self):
        """Calls build and saves the template matrix containing all traces. 
        Furthermore A stores in its last row the amount of ones per column to determine how many
        colored pixels each trace has. JUST DO THIS ONCE PER CONFIGURATION"""
        
        H = self.build_trace_matrix_coefficients_orders()
        H = H.tocsr()
        save_npz("H_matrix_flux_all_orders.npz", H)
        return
    

    ##################################################
    # Sensitivities, ones and zeros trace matrix with trace count, all orders
    ###################################################
    
    def build_trace_matrix_sensitivities_all_orders(self):
        """Builds the matrix A with size(x_pixel*y_pixel)*(x_pixel*y_pixel+1) where each row is a pixel in the dispersed image and each column 
        the trace at the object in direct coordinates. Last row includes how many trace pixels per column. Called by def build_and_save_matrix"""


        hdu = fits.open("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\SenseConfig\\wfss-grism-configuration\\NIRISS.GR150R.F150W.1.etc.1.5.2.sens.fits") #F150W, GR150R
        data1= hdu[1].data
        wavelength1 = data1["WAVELENGTH"]
        sensitivity1 = data1["SENSITIVITY"]
        #sensitivity1[sensitivity1<1e-9]=1e-9
        ####normalizing#################
        ######## if still not correct 
        #sens1_norm = np.divide(sensitivity1, sensitivity1, out=np.zeros_like(sensitivity1, dtype=float), where=sensitivity1!=0)
        #sensitivity1=sensitivity1/mean1 #normalized
        normalizer = np.max(sensitivity1)
        sens1_norm = sensitivity1/normalizer
        hdu.close()

        hdu = fits.open("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\SenseConfig\\wfss-grism-configuration\\NIRISS.GR150R.F150W.0.etc.1.5.2.sens.fits") #F150W, GR150R
        data0= hdu[1].data
        wavelength0 = data0["WAVELENGTH"]
        sensitivity0 = data0["SENSITIVITY"] 
        #sensitivity0[sensitivity0<1e-9]=1e-9
        #sens0_norm = np.divide(sensitivity0, sensitivity1, out=np.zeros_like(sensitivity1, dtype=float), where=sensitivity1!=0) #normalized by the same factor as 1st order
        sens0_norm = sensitivity0/normalizer
        hdu.close()

        hdu = fits.open("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\SenseConfig\\wfss-grism-configuration\\NIRISS.GR150R.F150W.2.etc.1.5.2.sens.fits") #F150W, GR150R
        data2= hdu[1].data
        wavelength2 = data2["WAVELENGTH"]
        sensitivity2 = data2["SENSITIVITY"] 
        #sensitivity2[sensitivity2<1e-9]=1e-9
        #sens2_norm = np.divide(sensitivity2, sensitivity1, out=np.zeros_like(sensitivity1, dtype=float), where=sensitivity1!=0) #normalized by the same factor as 1st order
        sens2_norm = sensitivity2/normalizer
        hdu.close()
        sens_interp = [interp1d(wavelength1, sens1_norm, bounds_error=False, fill_value=0.0),interp1d(wavelength0, sens0_norm, bounds_error=False, fill_value=0.0),interp1d(wavelength2, sens2_norm, bounds_error=False, fill_value=0.0)]
        delta_lambda = self.lambdas[1]-self.lambdas[0]

        order102 = ["A","B","C"]
                    #assembles matrix A
        N = self.x_pixel * self.y_pixel
        A=lil_matrix((N,N)) #good for sparse matrices
        for order in order102:
            if order == "A":
                senscount = 1
            elif order == "B":
                senscount = 0
            elif order == "C":
                senscount = 2
            print(order)
            for i in range (self.x_pixel): #iterating over all pixels
                    for j in range (self.y_pixel): 
                                    # grismagic expects source positions in detector coords
                        x0 = self.xmin + i
                        y0 = self.ymin + j
                        # Compute the trace offsets along the grism
                        # offset=None lets grismagic choose the full trace range
                        x_trace, y_trace, lam_trace = self.tr.get_trace(x0, y0, order=order)
                    
                        x_new = np.round(x_trace).astype(int) #trace in observation coordinates
                        y_new = np.round(y_trace).astype(int) 
                    
                        #only visible trace
                        mask = (x_new >= self.xmin) & (x_new < self.xmax) & (y_new >= self.ymin) & (y_new < self.ymax) 
                        
                        if not np.any(mask):
                            continue

                        x_valid = x_new[mask]
                        y_valid = y_new[mask]
                        lam_valid = lam_trace[mask]
                        
                        rows = (x_valid - self.xmin)*self.y_pixel +(y_valid-self.ymin)
                        k= i*self.y_pixel +j
                        
                        values = sens_interp[senscount](lam_valid).reshape(-1, 1)
                        for r, v in zip(rows, values):
                            A[r, k] += v #*delta_lambda
                
            # compute row sum to mark how many trace pixels per column
        row_sum = A.sum(axis = 0)
            
            # Append it as last row to A
        A = vstack([A, row_sum])
        return A
    
    def build_and_save_trace_matrix_sensitivities_all_orders(self):
        """Calls build and saves the template matrix containing all traces. 
        Furthermore A stores in its last row the amount of ones per column to determine how many
        colored pixels each trace has. JUST DO THIS ONCE PER CONFIGURATION"""
        A = self.build_trace_matrix_sensitivities_all_orders()
        A = A.tocsr()
        save_npz("A_F150W_20_500_matrix_with_trace_count_sensitivities_all_orders.npz", A)
        return
    #####################################
    # Matrix with basis for one order with PCA basis
    ####################################


    def build_trace_matrix_coefficients_PCA(self):
        """Builds the matrix H with size(x_pixel*y_pixel)*(x_pixel*y_pixel*h) where each row is a pixel in the dispersed image and each column 
        the trace at a basis function at the object in direct coordinates. Uses the spectrum function phi_m(lambda), s.t. we store phi in H and only need a_m(x,y) in d. 
        Called by def build_and_save_matrix
            f = H @ a_tilde
            where a_tilde consists out of a_0(k_1),a_1(k_1)....a_n(k_last)
        """
        #assembles matrix H
        N = self.x_pixel * self.y_pixel # row dimension of H
        p,q = self.df.shape
        h=q-1 # amount of basis functions. -1 because 1st column are the wavelengths.
   
        H=lil_matrix((N,N*h)) #good for sparse matrices
        Phi = self.eigenspectra_basis()
        

        for i in range(self.x_pixel):
            for j in range(self.y_pixel):
                k= i*self.y_pixel +j #gets column index right, for which pixel/column are we inserting the trace
                
                x0 = i + self.xmin
                y0 = j + self.ymin

                try:
                    # CRUCIAL: align wavelengths with your basis
                    x_trace, y_trace = self.tr.get_trace_at_wavelength(x0, y0, order="A", lam=self.lambdas)
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
                lam_indices = np.where(mask)[0]
                
                rows = (x_valid - self.xmin)*self.y_pixel +(y_valid-self.ymin) # gets row index right for trace. In which rows appears the trace?
        
            
                for idx, row in enumerate(rows): # enumerates gives back the index of the row and the row at the same time.
                    l_indx = lam_indices[idx]
                    for m in range(h):
                        col = k*h+ m #correct column indexing
                        H[row,col] += Phi[l_indx,m] # instead of a trace 00111110000 we add now the function phi(lambdas) to positions of the trace
            
        return H
    
    def build_and_save_trace_matrix_coefficients_PCA(self):
        """Calls build and saves the template matrix containing all traces. 
        Furthermore A stores in its last row the amount of ones per column to determine how many
        colored pixels each trace has. JUST DO THIS ONCE PER CONFIGURATION"""
        
        H = self.build_trace_matrix_coefficients_PCA()
        H = H.tocsr()
        save_npz("H_matrix_flux_1st_order_PCA.npz", H)
        return
    
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
        delta_lambda = self.lambdas[1] - self.lambdas[0]
        
        order102 = ["A","B","C"]
        for order in order102:
            if order == "A":
                senscount = 1
                Phi = Phi * sens_interp[senscount](self.lambdas)[:,None]
            elif order == "B":
                senscount = 0
                Phi = Phi * sens_interp[senscount](self.lambdas)[:,None]
            elif order == "C":
                senscount = 2
                Phi = Phi * sens_interp[senscount](self.lambdas)[:,None]
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

                    lam_valid = self.lambdas[lam_idx]
                    values = sens_interp[senscount](lam_valid).reshape(-1, 1)
               
                    for idx, row in enumerate(rows): # enumerates gives back the index of the row and the row at the same time.
                        l_indx = lam_idx[idx]
                        sens_idx = values[idx]
                        
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
        save_npz("H_matrix_F150W_flux_2048_2048_orders_PCA_sensitivity.npz", H)
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
    