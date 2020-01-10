import numba as nb
import numpy as np
import math
import sklearn.decomposition as sk
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy.special import gammainc
from scipy.special import lambertw
from matplotlib import pyplot as plt
import scipy.io
from functools import lru_cache

@nb.njit
def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros((len(X[0,:]),len(bins)))
    for j in range(len(map_to_bins[0,:])):
        for i in map_to_bins[:,j]:
            r[j,i-1] += 1
    return r

def randsphere(n_points,ndim,radius,center = []):
    if center == []:
        center = np.array([0]*ndim)
    r = radius
    x = np.random.normal(size=(n_points, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_points,1),(1,ndim))
    p = center + np.multiply(x,frtiled)
    return p, center

def preprocessing(X,center,dimred,whiten,projectonsphere,ConditionalNumber = 10,ncomp=1):
    '''
    %preprocessing form preprocessed dataset
    %
    %Inputs
    %   X is n-by-d data matrix with n d-dimensional datapoints.
    %   center is boolean. True means subtraction of mean vector.
    %   dimred is boolean. True means applying of dimensionality reduction with
    %       PCA. Number of used PCs is defined by ConditionalNumber argument.
    %   whiten is boolean. True means applying of whitenning. True whiten
    %       automatically caused true dimred.
    %   projectonsphere is boolean. True means projecting data onto unit sphere
    %   varargin contains Name Value pairs. One possible value can be:
    %       'ConditionalNumber' - a positive real value used to select the top
    %           principal components. We consider only PCs with eigen values
    %           which are not less than the maximal eigenvalue divided by
    %           ConditionalNumber Default value is 10. 
    %
    %Outputs:
    %   X is preprocessed data matrix.'''
    
    #centering
    nobjects = len(X[:,0])
    sampleMean = np.mean(X,axis=0)
    if center:
        X = X-sampleMean
    #dimensionality reduction if requested dimensionality reduction or whitening
    PCAcomputed = 0
    if dimred or whiten:
        pca = sk.PCA()
        u = pca.fit_transform(X)
        v = pca.components_.T
        s = pca.explained_variance_
        PCAcomputed = 1
        sc = s/s[0]
        ind = np.where(sc > 1/ConditionalNumber)[0]
        X = X @ v[:,ind]
        if ncomp:
            print('%i components are retained using factor %2.2f' %(len(ind),ConditionalNumber))

    #whitening
    if whiten:
        X = u[:,ind]
        st = np.std(X,axis=0,ddof=1)
        X = X/st
    # #project on sphere (scale each vector to unit length)
    if projectonsphere:
        st = np.sqrt(np.sum(X**2,axis=1))
        st = np.array([st]).T
        X = X/st
    
    return X    

@lru_cache()
def probability_unseparable_sphere(alpha,n):
    ''' 
    %probability_unseparable_sphere calculate theoretical probability for point
    %to be inseparable for dimension n
    %
    %Inputs:
    %   alpha is 1-by-d vector of possible alphas. Must be row vector or scalar
    %   n is c-by-1 vector of dimnesions. Must be column vector or scalar.
    %
    %Outputs:
    %   p is c-by-d matrix of probabilities.'''
    p = np.power((1-np.power(alpha,2)),(n-1)/2)/(alpha*np.sqrt(2*np.pi*n))
    return p

def checkSeparability(xy,alpha):
    dxy = np.diag(xy)
    sm = (xy/dxy).T
    sm = sm - np.diag(np.diag(sm))
    sm = sm>alpha
    py = sum(sm.T)
    py = py/len(py[0,:])
    separ_fraction = sum(py==0)/len(py[0,:])
    
    return separ_fraction,py

def checkSeparabilityMultipleAlpha(data,alpha):
    '''%checkSeparabilityMultipleAlpha calculate fraction of points inseparable
    %for each alpha and fraction of points which are inseparable from each
    %point for different alpha.
    %
    %Inputs:
    %   data is data matrix to calculate separability. Each row contains one
    %       data point.
    %   alpha is array of alphas to test separability.
    %
    %Outputs:
    %   separ_fraction fraction of points inseparable from at least one point.
    %       Fraction is calculated for each alpha.
    %   py is n-by-m matrix. py(i,j) is fraction of points which are
    %       inseparable from point data(i, :) for alphas(j).'''


    #Number of points per 1 loop. 20k assumes approx 3.2GB
    nP = 2000

    #Normalize alphas
    if len(alpha[:,0])>1:
        alpha = alpha.T
    addedone = 0
    if max(alpha[0,:])<1:
        alpha = np.array([np.append(alpha,1)])
        addedone = 1

    alpha = np.concatenate([[float('-inf')],alpha[0,:], [float('inf')]])

    n = len(data)
    counts = np.zeros((n, len(alpha)))
    leng = np.zeros((n, 1))
    for k in range(0,n,nP):
        #print('Chunk +{}'.format(k))
        e = k + nP 
        if e > n:
            e = n
        # Calculate diagonal part, divide each row by diagonal element
        xy = data[k:e, :] @ data[k:e, :].T
        leng[k:e] = np.diag(xy)[:,None]
        xy = xy - np.diag(leng[k:e].squeeze())
        xy = xy / leng[k:e]
        counts[k:e, :] = counts[k:e, :] + histc(xy.T, alpha)
        # Calculate nondiagonal part
        for kk in range(0,n,nP):
            #Ignore diagonal part
            if k == kk:
                continue                         
            ee = kk + nP 
            if ee > n:
                ee = n

            xy = data[k:e, :] @ data[kk:ee, :].T
            xy = xy / leng[k:e]
            counts[k:e, :] = counts[k:e, :] + histc(xy.T, alpha)

    #Calculate cumulative sum
    counts = np.cumsum(counts[:,::-1],axis=1)[:,::-1]
    
    #print(counts)

    py = counts/(n-1)
    py = py.T
    if addedone:
        py = py[1:-2,:]
    else:
        py = py[1:-1,:]

    separ_fraction = sum(py==0)/len(py[0,:])
    
    return separ_fraction, py


def dimension_uniform_sphere(py,alphas):
    '''
    %Gives an estimation of the dimension of uniformly sampled n-sphere
    %corresponding to the average probability of being unseparable and a margin
    %value 
    %
    %Inputs:
    %   py - average fraction of data points which are INseparable.
    %   alphas - set of values (margins), must be in the range (0;1)
    % It is assumed that the length of py and alpha vectors must be of the
    % same.
    %
    %Outputs:
    %   n - effective dimension profile as a function of alpha
    %   n_single_estimate - a single estimate for the effective dimension 
    %   alfa_single_estimate is alpha for n_single_estimate.
    '''
    
    if len(py)!=len(alphas[0,:]):
        raise ValueError('length of py (%i) and alpha (%i) does not match'%(len(py),len(alphas[0,:])))
    
    if np.sum(alphas <= 0) > 0 or np.sum(alphas >= 1) > 0:
        raise ValueError(['"Alphas" must be a real vector, with alpha range, the values must be within (0,1) interval'])

    #Calculate dimension for each alpha
    n = np.zeros((len(alphas[0,:])))
    for i in range(len(alphas[0,:])):
        if py[i] == 0:
            #All points are separable. Nothing to do and not interesting
            n[i]=np.nan
        else:
            p  = py[i]
            a2 = alphas[0,i]**2
            w = np.log(1-a2)
            n[i] = lambertw(-(w/(2*np.pi*p*p*a2*(1-a2))))/(-w)
 
    n[n==np.inf] = float('nan')
    #Find indices of alphas which are not completely separable 
    inds = np.where(~np.isnan(n))[0]
    #Find the maximal value of such alpha
    alpha_max = max(alphas[0,inds])
    #The reference alpha is the closest to 90 of maximal partially separable alpha
    alpha_ref = alpha_max*0.9
    k = np.where(abs(alphas[0,inds]-alpha_ref)==min(abs(alphas[0,:]-alpha_ref)))[0]
    #Get corresponding values
    alfa_single_estimate = alphas[0,inds[k]]
    n_single_estimate = n[inds[k]]
    
    return n,n_single_estimate,alfa_single_estimate

def SeparabilityAnalysis(X,ConditionalNumber=10,ProjectOnSphere = 1,alphas = np.array([np.arange(.6,1,.02)]),ProducePlots = 1,ncomp = 0):
    '''
    %Performs standard analysis of separability and produces standard plots. 
    %
    %Inputs:
    %   X  - is a data matrix with one data point in each row.
    %   Optional arguments in varargin form Name, Value pairs. Possible names:
    %       'ConditionalNumber' - a positive real value used to select the top
    %           princinpal components. We consider only PCs with eigen values
    %           which are not less than the maximal eigenvalue divided by
    %           ConditionalNumber Default value is 10.
    %       'ProjectOnSphere' - a boolean value indicating if projecting on a
    %           sphere should be performed. Default value is true.
    %       'Alphas' - a real vector, with alpha range, the values must be given increasing
    %           within (0,1) interval. Default is [0.6,0.62,...,0.98].
    %       'ProducePlots' - a boolean value indicating if the standard plots
    %           needs to be drawn. Default is true.
    %       'ncomp' - whether to print number of retained principal components
    %Outputs:
    %   n_alpha - effective dimension profile as a function of alpha
    %   n_single - a single estimate for the effective dimension 
    %   p_alpha - distributions as a function of alpha, matrix with columns
    %       corresponding to the alpha values, and with rows corresponding to
    %       objects. 
    %   separable_fraction - separable fraction of data points as a function of
    %       alpha
    %   alphas - alpha values
    '''
    npoints = len(X[:,0])
    # Preprocess data
    Xp = preprocessing(X,1,1,1,ProjectOnSphere,ConditionalNumber=ConditionalNumber,ncomp=ncomp)
    # Check separability
    [separable_fraction,p_alpha] = checkSeparabilityMultipleAlpha(Xp,alphas)    
    # Calculate mean of fraction of separable points for each alpha.
    py_mean = np.mean(p_alpha,axis=1)    
    [n_alpha,n_single,alpha_single] = dimension_uniform_sphere(py_mean,alphas)

    alpha_ind_selected = np.where(n_single==n_alpha)[0]
    
    if ProducePlots:
        #Define the minimal and maximal dimensions for theoretical graph with
        # two dimensions in each side
        n_min = np.floor(min(n_alpha))-2;
        n_max = np.floor(max(n_alpha)+0.8)+2;
        if n_min<1:
            n_min = 1

        ns = np.arange(n_min,n_max+1)
        
        plt.figure()
        plt.plot(alphas[0,:],n_alpha,'ko-');plt.plot(alphas[0,alpha_ind_selected],n_single,'rx',markersize=16)
        plt.xlabel('\u03B1',fontsize=16); plt.ylabel('Effective dimension',fontsize=16) ; locs, labels = plt.xticks(); plt.show()
        nbins = int(round(np.floor(npoints/200)))

        if nbins<20:
            nbins = 20
        

        plt.figure()
        plt.hist(p_alpha[alpha_ind_selected,:][0],bins=nbins)
        plt.xlabel('inseparability prob.p for \u03B1=%2.2f'%(alphas[0,alpha_ind_selected]),fontsize=16); plt.ylabel('Number of values');plt.show()

        plt.figure()
        plt.xticks(locs,labels);
        pteor = np.zeros((len(ns),len(alphas[0,:])))
        for k in range(len(ns)):
            for j in range(len(alphas[0,:])):
                pteor[k,j] = probability_unseparable_sphere(alphas[0,j],ns[k])

        for i in range(len(pteor[:,0])):
            plt.semilogy(alphas[0,:],pteor[i,:],'-',color='r')
        plt.xlim(min(alphas[0,:]),1)
        if True in np.isnan(n_alpha):
            plt.semilogy(alphas[0,:np.where(np.isnan(n_alpha))[0][0]],py_mean[:np.where(np.isnan(n_alpha))[0][0]],'bo-','LineWidth',3);
        else: 
            plt.semilogy(alphas[0,:],py_mean,'bo-','LineWidth',3);

        plt.xlabel('\u03B1'); plt.ylabel('Mean inseparability prob.',fontsize=16);
        plt.title('Theor.curves for n=%i:%i'%(n_min,n_max))
        plt.show()


    return n_alpha,n_single,p_alpha,alphas,separable_fraction,Xp
