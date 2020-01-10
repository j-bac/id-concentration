#Original Author: Francesco Mottes
#https://github.com/fmottes/TWO-NN
#MIT License
#
#Copyright (c) 2019 fmottes
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#Author of the speed modifications (with sklearn dependencies): Jonathan Bac
#Date  : 02-Jan-2020
#-----------------------------

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_chunked
from ._commonfuncs import get_nn

def twonn(data, return_xy=False):
    """
    Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.
    
    -----------
    Parameters:
    
    data : 2d array-like
        2d data matrix. Samples on rows and features on columns.
    return_xy : bool (default=False)
        Whether to return also the coordinate vectors used for the linear fit.
        
    -----------
    Returns:
    
    d : int
        Intrinsic dimension of the dataset according to TWO-NN.
    x : 1d array (optional)
        Array with the -log(mu) values.
    y : 1d array (optional)
        Array with the -log(F(mu_{sigma(i)})) values.
        
    -----------
    References:
    
    [1] E. Facco, M. d’Errico, A. Rodriguez & A. Laio
        Estimating the intrinsic dimension of datasets by a minimal neighborhood information (https://doi.org/10.1038/s41598-017-11873-y)
    
    
    """
    
    
    data = np.array(data)
    
    N = len(data)
    
    ### OLD CODE ###
    #mu = []
    #for i,x in enumerate(data):
    #    
    #    dist = np.sort(np.sqrt(np.sum((x-data)**2, axis=1)))
    #    r1, r2 = dist[dist>0][:2]
    #
    #    mu.append((i+1,r2/r1))
    
    #mu = r2/r1 for each data point
    if data.shape[1] > 25: #relatively high dimensional data, use distance matrix generator
        distmat_chunks = pairwise_distances_chunked(data)
        _mu = np.zeros((len(data)))
        i = 0
        for x in distmat_chunks:
            x = np.sort(x,axis=1)
            r1, r2 = x[:,1], x[:,2]
            _mu[i:i+len(x)] = (r2/r1)
            i += len(x)
        mu = list(zip(np.arange(1,N+1), _mu))
        
    else: #relatively low dimensional data, search nearest neighbors directly
        dists, _ = get_nn(data,k=2)
        r1,r2 = dists[:,0],dists[:,1]
        mu = list(zip(np.arange(1,N+1),(r2/r1)))


    #permutation function
    sigma_i = dict(zip(range(1,len(mu)+1), np.array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))

    mu = dict(mu)

    #cdf F(mu_{sigma(i)})
    F_i = {}
    for i in mu:
        F_i[sigma_i[i]] = i/N

    #fitting coordinates
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1-F_i[i] for i in sorted(mu.keys())])

    #avoid having log(0)
    x = x[y>0]
    y = y[y>0]

    y = -1*np.log(y)

    #fit line through origin to get the dimension
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
        
    if return_xy:
        return d, x, y
    else: 
        return d
