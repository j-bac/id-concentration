### Credits to Gabriele Lombardi
### https://fr.mathworks.com/matlabcentral/fileexchange/40112-intrinsic-dimensionality-estimation-techniques
### for the original MATLAB implementation

### Credits to Kerstin Johnsson
### https://cran.r-project.org/web/packages/intrinsicDimension/index.html
### for the R implementation

import sys
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
from scipy.special import i0,i1,digamma,gammainc
from scipy.interpolate import interp1d,interp2d
from ._commonfuncs import binom_coeff, get_nn, randsphere, lens, indnComb
from pathlib import Path
path_to_estimators = str(Path(__file__).resolve().parent)


def KL(nocal, caldat, k):
    kld = KLd(nocal['dhat'], caldat['dhat'], k)
    klnutau = KLnutau(nocal['mu_nu'], caldat['mu_nu'],
                     nocal['mu_tau'], caldat['mu_tau'])
    #print(klnutau)
    return(kld + klnutau)

def KLd(dhat, dcal, k):
    H_k = np.sum(1/np.arange(1,k+1))    
    quo = dcal/dhat
    a = np.power(-1,np.arange(k+1))*np.array(list(binom_coeff(k,i) for i in range(k+1)))*digamma(1 + np.arange(k+1)/quo)
    return(H_k*quo - np.log(quo) - (k-1)*np.sum(a))

def KLnutau(nu1, nu2, tau1, tau2):
    return(np.log(min(sys.float_info.max,i0(tau2))/min(sys.float_info.max,i0(tau1))) + 
        min(sys.float_info.max,i1(tau1))/min(sys.float_info.max,i0(tau1))*(tau1 - tau2*np.cos(nu1-nu2)))

def nlld(d, rhos, k, N):
    return(-lld(d, rhos, k, N))

def lld(d, rhos, k, N):
    if (d == 0):
        return(np.array([-1e30]))
    else:
        return N*np.log(k*d) + (d-1)*np.sum(np.log(rhos)) + (k-1)*np.sum(np.log(1-rhos**d))
    
def nlld_gr(d, rhos, k, N):
    if (d == 0):
        return(np.array([-1e30]))
    else:
        return -(N/d + np.sum(np.log(rhos) - (k-1)*(rhos**d)*np.log(rhos)/(1 - rhos**d)))

def MIND_MLk(rhos, k, D):
    N = len(rhos)
    d_lik = np.array([np.nan]*D)
    for d in range(D):
        d_lik[d] = lld(d, rhos, k, N)
    return(np.argmax(d_lik))

def MIND_MLi(rhos, k, D, dinit):
    res = minimize(fun=nlld,
            x0=np.array([dinit]),
            jac=nlld_gr,
            args=(rhos, k, len(rhos)),
            method = 'L-BFGS-B',
            bounds=[(0,D)])

    #if(!is.null(res$message)) print(res$message)
    return(res['x'])  


def MIND_MLx(data, k, D, ver):
    nbh_data,idx = get_nn(data, k+1)
    rhos = nbh_data[:,0]/nbh_data[:,-1]
    
    d_MIND_MLk = MIND_MLk(rhos, k, D)
    if (ver == 'MIND_MLk'):
        return(d_MIND_MLk)

    d_MIND_MLi = MIND_MLi(rhos, k, D, d_MIND_MLk)
    if (ver == 'MIND_MLi'):
        return(d_MIND_MLi)
    else:
        raise ValueError("Unknown version: ", ver)

def Ainv(eta):
    if (eta < .53):
        return(2*eta + eta**3 + 5*(eta**5)/6)
    elif (eta < .85):
        return(-.4 + 1.39*eta + .43/(1-eta))
    else:
        return(1/((eta**3)-4*(eta**2)+3*eta))

def loc_angles(pt, nbs):
    vec = nbs-pt
   # if(len(pt) == 1):
   #     vec = vec.T
    vec_len = lens(vec)
    combs = indnComb(len(nbs), 2).T
    sc_prod = np.sum(vec[combs[0,:]]*vec[combs[1,:]],axis=1)
    #if (length(pt) == 1) {
    #print(sc.prod)
    #print((vec.len[combs[1, ]]*vec.len[combs[2, ]]))
    #}
    cos_th = sc_prod/(vec_len[combs[0,:]]*vec_len[combs[1,:]])
    if (any(abs(cos_th) > 1)):
        print(cos_th[np.abs(cos_th) > 1])
    return(np.arccos(cos_th))

def angles(data, nbs):
    N = len(data)
    k = nbs.shape[1]
    
    thetas = np.zeros((N, binom_coeff(k, 2)))
    for i in range(N):
        nb_data = data[nbs[i, ],]
        thetas[i, ] = loc_angles(data[i, ], nb_data)    
    return(thetas)

def ML_VM(thetas):
    sinth = np.sin(thetas)
    costh = np.cos(thetas)
    nu = np.arctan(np.sum(sinth)/np.sum(costh))
    eta = np.sqrt(np.mean(costh)**2 + np.mean(sinth)**2)
    tau = Ainv(eta)
    return dict(nu = nu, tau = tau)


def dancoDimEstNoCalibration(data, k, D, n_jobs=1):
    nbh_data,idx = get_nn(data, k+1,n_jobs=n_jobs)
    rhos = nbh_data[:,0]/nbh_data[:,-1]
    d_MIND_MLk = MIND_MLk(rhos, k, D)
    d_MIND_MLi = MIND_MLi(rhos, k, D, d_MIND_MLk)

    thetas = angles(data, idx[:,:k])
    ml_vm = list(map(ML_VM,thetas))
    mu_nu = np.mean([i['nu'] for i in ml_vm])
    mu_tau = np.mean([i['tau'] for i in ml_vm])
    if(data.shape[1] == 1):
        mu_tau = 1
        
    return dict(dhat = d_MIND_MLi, mu_nu = mu_nu, mu_tau = mu_tau)

def DancoCalibrationData(k, N):
    me = dict(k = k,
            N = N,
            calibration_data = list(),
            maxdim = 0)    
    return(me)

def increaseMaxDimByOne(dancoCalDat):
    newdim = dancoCalDat['maxdim'] + 1
    MIND_MLx_maxdim = newdim*2+5
    dancoCalDat['calibration_data'].append(dancoDimEstNoCalibration(randsphere(dancoCalDat['N'], newdim,1,center=[0]*newdim)[0], 
                                                                         dancoCalDat['k'], 
                                                                         MIND_MLx_maxdim))
    dancoCalDat['maxdim'] = newdim
    return(dancoCalDat)
        
def increaseMaxDimByOne_precomputedSpline(dancoCalDat,DANCo_splines):
    newdim = dancoCalDat['maxdim'] + 1
    dancoCalDat['calibration_data'].append({'dhat':DANCo_splines['spline_dhat'](newdim,dancoCalDat['N']),
                                            'mu_nu':DANCo_splines['spline_mu'](newdim,dancoCalDat['N']),
                                            'mu_tau':DANCo_splines['spline_tau'](newdim,dancoCalDat['N'])})
    dancoCalDat['maxdim'] = newdim
    return(dancoCalDat)

def computeDANCoCalibrationData(k,N,D):
    cal=DancoCalibrationData(k,N)
    while (cal['maxdim'] < D):
        if cal['maxdim']%10==0:
            print(cal['maxdim'])
        cal = increaseMaxDimByOne(cal)
    return cal


def dancoDimEst(data, k, D, ver = 'DANCo', fractal = True, calibration_data = None):
    
    cal = calibration_data
    N = len(data)
    
    if cal is not None:
        if (cal['k'] != k):
            raise ValueError("Neighborhood parameter k = %s does not agree with neighborhood parameter of calibration data, cal$k = %s",
                   k, cal['k'])
        if (cal['N'] != N):
            raise ValueError("Number of data points N = %s does not agree with number of data points of calibration data, cal$N = %s",
                   N, cal['N'])
  
    if (ver != 'DANCo' and ver != 'DANCoFit'):
        return(MIND_MLx(data, k, D, ver))
  
    nocal = dancoDimEstNoCalibration(data, k, D)
    if any(np.isnan(val) for val in nocal.values()):
        return dict(de=np.nan, kl_divergence = np.nan, calibration_data=cal)

    if (cal is None):
        cal = DancoCalibrationData(k, N)

    if (cal['maxdim'] < D): 
        
        if ver == 'DANCoFit':
            print("Generating DANCo calibration data from precomputed spline interpolation for cardinality 50 to 5000, k = 10, dimensions 1 to 100")

            #load precomputed splines as a function of dimension and dataset cardinality
            DANCo_splines = {}
            for spl in ['spline_dhat','spline_mu','spline_tau']:
                with open(path_to_estimators+'/DANCoFit/DANCo_'+spl+'.pkl', 'rb') as f:
                    DANCo_splines[spl]=pickle.load(f)
            #compute interpolated statistics
            while (cal['maxdim'] < D):
                cal = increaseMaxDimByOne_precomputedSpline(cal,DANCo_splines)
    
        else:
            print("Computing DANCo calibration data for N = {}, k = {} for dimensions {} to {}".format(N, k, cal['maxdim']+1, D))
            
            #compute statistics
            while (cal['maxdim'] < D):
                cal = increaseMaxDimByOne(cal)
        

    kl = np.array([np.nan]*D) 
    for d in range(D) :
        kl[d] = KL(nocal, cal['calibration_data'][d], k) 

    de = np.argmin(kl)+1
    
    if fractal:
        # Fitting with a cubic smoothing spline:
        f=interp1d(np.arange(1,D+1),kl,kind='cubic')
        # Locating the minima:
        de_fractal=minimize(f, de, bounds=[(1,D+1)],tol=1e-3)['x']
        return dict(de=de_fractal, kl_divergence = kl[de-1], calibration_data = cal)
    else:
        return dict(de=de, kl_divergence = kl[de-1], calibration_data = cal)
