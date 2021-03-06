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
from sklearn.linear_model import LinearRegression
from ._commonfuncs import get_nn

def twonn(data, return_xy=False, discard_fraction = 0.1, dist = False):
    """
    Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.
    
    -----------
    Parameters:
    
    data : 2d array-like
        2d data matrix. Samples on rows and features on columns.
    return_xy : bool (default=False)
        Whether to return also the coordinate vectors used for the linear fit.
    discard_fraction : float between 0 and 1
        Fraction of largest distances to discard (heuristic from the paper)
    dist : bool (default=False)
        Whether data is a precomputed distance matrix
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
    
    if dist:
        r1,r2 = dists[:,0],dists[:,1]
        _mu = r2/r1
        mu = _mu[np.argsort(_mu)[:int(N*(1-discard_fraction))]] #discard the largest distances
    
    else:    
        # mu = r2/r1 for each data point
        if data.shape[1] > 25: #relatively high dimensional data, use distance matrix generator
            distmat_chunks = pairwise_distances_chunked(data)
            _mu = np.zeros((len(data)))
            i = 0
            for x in distmat_chunks:
                x = np.sort(x,axis=1)
                r1, r2 = x[:,1], x[:,2]
                _mu[i:i+len(x)] = (r2/r1)
                i += len(x)

            mu = _mu[np.argsort(_mu)[:int(N*(1-discard_fraction))]] #discard the largest distances

        else: #relatively low dimensional data, search nearest neighbors directly
            dists, _ = get_nn(data,k=2)
            r1,r2 = dists[:,0],dists[:,1]
            _mu = r2/r1
            mu = _mu[np.argsort(_mu)[:int(N*(1-discard_fraction))]] #discard the largest distances

    # Empirical cumulate
    Femp = np.arange(int(N*(1-discard_fraction)))/N

    # Fit line
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))

    d = lr.coef_[0][0] # extract slope
        
    if return_xy:
        return d, x, y
    else: 
        return d
