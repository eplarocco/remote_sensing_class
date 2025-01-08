import matplotlib.pyplot as plt
import scripts_M2 as sm2
import numpy as np
import time


class GaussianClassification():
    def __init__(self, im): 
        self.im = im
        self.wl = np.asarray(im.bands.centers)
        self.imArr = im.Arr  
        self.nrows = im.Arr.shape[0]
        self.ncols = im.Arr.shape[1]
        self.nbands = im.Arr.shape[2]
        self.gt_im = np.zeros((im.nrows,im.ncols))
        
        
    def load_gt(self, fname, verbose=False):
        if verbose: 
            print('Loading groundtruth...')
            start_time = time.time()        
        ### Start Code ###
        file1 = open(fname, 'r')
        Lines = file1.readlines()

        # Strips the newline character
        self.groundtruth_classes = {}
        self.class_names = []

        classIndex = 0
        idx = 0
        x = 0
        for line in Lines:
            
            if line[0]==';':
                line = line.strip()
                #  reading the header info
                if line[:11] == '; ROI name:':
                    name = line[12:]
                    self.class_names.append(name)
                    self.groundtruth_classes[name] = {'classIndex': classIndex, 'locations': []}
                    classIndex = classIndex + 1
                if line[:11] == 'ROI npts:':
                    self.groundtruth_classes[name]['npts'] = float(line[12:])
                    
            else:
                line = line.strip()
                # reading the data
                if len(line)==0:
                    # blank line - swtch to next class
                    idx = idx + 1
                else:
                    loc = line.split(' ') # read the locations, split by spaces
                    loc = [i for i in loc if i != '']  # remove blank spaces
                    loc = [loc[2],loc[1]] # get the x and y coords
                    self.groundtruth_classes[self.class_names[idx]]['locations'].append(loc)

        # Iterate throught the classes and color the ground truth image for each pixel in the ground truth:
        for key in self.groundtruth_classes.keys():
            idx = self.groundtruth_classes[key]['classIndex']
            locations = self.groundtruth_classes[key]['locations']
            for x,y in locations:
                self.gt_im[int(x),int(y)] = int(idx)+1

        self.gt_list = np.reshape(self.gt_im, (self.im.nrows*self.im.ncols))
        ### End Code ###
        if verbose:
            print(f'Total time: {(time.time()-start_time):.2f} seconds')
    
    def train(self, tol=10**(-8), verbose=False):
        if verbose: 
            print('Training...')
            start_time = time.time()        
        ### Start Code ###
        self.nClasses = len(self.groundtruth_classes.keys())

        self.class_means = np.zeros((self.nClasses, self.im.nbands))
        class_covariances = np.zeros((self.nClasses, self.im.nbands, self.im.nbands))
        self.C = np.zeros((self.im.nbands,self.im.nbands))

        class_indices = np.unique(self.gt_list).astype(int)
        for idx in class_indices:
            # skip idx==0 because that corresponds to the unlabeled (background) class
            if idx > 0: 
                class_locations = np.where(self.gt_list==idx)[0]
                num_class_spectra = len(class_locations) # N_i
                class_spectra = self.im.List[class_locations,:]
                self.class_means[idx-1,:] = np.mean(class_spectra, axis=0) # \mu_i
                class_covariances[idx-1,:,:] = np.cov(class_spectra.T) # \Sigma_i
                self.C = self.C + num_class_spectra*class_covariances[idx-1,:,:]

        num_labeled_spectra = np.sum(self.gt_list>0)
        self.C = self.C/num_labeled_spectra  # \Sigma

        evals,evecs = np.linalg.eig(self.C) #eigenvalues = D and eigenvectors
        # truncate the small eigenvalues to stablize the inverse
        evals[evals<tol] = tol
        DiagMatrix = np.diag(evals**(-1/2)) #D^-1/2
        self.W = np.matmul(evecs,DiagMatrix) #P (W = PD)

        self.WimList = np.matmul(self.W.T, self.im.List.T).T #Y = W^T(X-mu) -> W^TX - W^Tmu

        ### End Code ###       
        if verbose: 
            print(f'Total time: {(time.time()-start_time):.2f} seconds')
        
    def predict(self, verbose=False):
        if verbose: 
            print('Predicting...')
            start_time = time.time()        
        ### Start Code ###
        # Compute Mahalanobis Distance to mean for each class, for all pixels
        self.MD_all = np.zeros((self.im.nrows*self.im.ncols, self.nClasses))
        for class_idx in range(self.nClasses):    
            # demean each pixel
            mu = self.class_means[class_idx,:]
            # whiten the mean
            Wmu = np.matmul(self.W.T, mu).T
            # subtract whitened mean from whitened data
            WimList_demean = self.WimList-Wmu
            # compute Mahalanobis Distance
            MDs = np.sum(WimList_demean**2, axis=1)
            MD = np.sqrt(MDs)
            # classify by minimum Mahalanobis distance
            self.MD_all[:,class_idx] = MD

        LDA_class_list = np.argmin(self.MD_all, axis=1)
        self.LDA_class_image = np.reshape(LDA_class_list, (self.im.nrows,self.im.ncols))

        ### End Code ###       
        if verbose: 
            print(f'Total time: {(time.time()-start_time):.2f} seconds')


    # Visualizations and Plots

    def plot_RGB(self, ):
        sm2.display_RGB(self.im.Arr, self.wl, rotate=True)
        plt.title('RGB Image');
        
    def plot_gt(self):
        plt.figure(figsize=(15,5)) 
        plt.imshow(np.flip(np.rot90(self.gt_im), axis=0), cmap='jet');
        plt.gca().invert_yaxis()  
        plt.xlabel('Row');
        plt.ylabel('Column');
        plt.title('Ground Truth Image')
    
    def plt_RGB_classes(self):
        class_only_image = np.zeros((self.nrows, self.ncols, self.nbands))
        for class_name in self.groundtruth_classes.keys():
            for x,y in self.groundtruth_classes[class_name]['locations']:
                class_only_image[int(x), int(y), :] = self.im.Arr[int(x), int(y), :]

        sm2.display_RGB(class_only_image, self.wl, stretch_pct=[0,99], rotate=True)
        plt.title('RGB Image');
        
    def plot_classification_results(self):
        plt.figure(figsize=(15,5)) 
        plt.imshow(np.flip(np.rot90(self.LDA_class_image), axis=0), cmap='jet');
        plt.gca().invert_yaxis()  
        plt.xlabel('Row');
        plt.ylabel('Column');
        plt.title('LDA Class Predictio n Image')
    
    def plt_class_means(self):
        plt.figure(figsize=(12,4))
        for i in range(self.nClasses):
            plt.plot(self.wl, self.class_means[i,:], label=self.class_names[i])
        plt.grid(True)
        plt.legend()
    
    def plot_class_sideBySide(self, fs=10):
        plt.figure(figsize=(fs,fs)) 
        plt.subplot(1,2,1)
        plt.imshow(self.gt_im, cmap='jet');
        plt.xlabel('Row');
        plt.ylabel('Column');
        plt.title('Ground Truth Image')
        plt.subplot(1,2,2)
        plt.imshow(self.LDA_class_image, cmap='jet');
        plt.xlabel('Row');
        plt.ylabel('Column');
        plt.title('LDA Class Predictio n Image')
        plt.tight_layout()
    
    def plt_scatter(self, b1=40, b2=150):
        # Create a scatterplot of the data
        plt.figure(figsize=(15,10))
        plt.scatter(self.im.List[:,b1], self.im.List[:,b2], s=5, alpha=0.5);
        plt.grid(True)
        plt.xlabel(f'Reflectance at {self.wl[b1]:.1f} nm')
        plt.ylabel(f'Reflectance at {self.wl[b2]:.1f} nm')
        plt.title('Scatterplot of Image Spectra');
        
    def plt_scatter_gt(self, b1=40, b2=150):
        # Create a scatterplot of the data, colored by ground truth class
        plt.figure(figsize=(15,10))
        bk_indiexes = np.where(self.gt_list==0)[0] # get hte indices only for pixels in ground truth classes
        plt.scatter(self.im.List[bk_indiexes,b1], self.im.List[bk_indiexes,b2], c='grey', s=3, alpha=0.5);
        gt_indiexes = np.where(self.gt_list>0)[0] # get hte indices only for pixels in ground truth classes
        plt.scatter(self.im.List[gt_indiexes,b1], self.im.List[gt_indiexes,b2], c=self.gt_list[gt_indiexes], s=3, alpha=0.5, cmap='jet');
        plt.grid(True)
        plt.xlabel(f'Reflectance at {self.wl[b1]:.1f} nm')
        plt.ylabel(f'Reflectance at {self.wl[b2]:.1f} nm')
        plt.title('Scatterplot of Image Spectra');
    
        
        
        
        
        
        