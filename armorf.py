import numpy as np
from numpy.linalg import inv
from numpy.linalg import cholesky

def armorf(x: np.ndarray, Nr = 1, Nl = 10, p = 2)->tuple:
    '''
    AR parameter estimation via LWR method by Morf modified
    args:
        x(L, N): np.ndarray, L: number of variables, N: number of samples(time points);
        Nr: int, number of realizations;
        Nl: int, length of every realization;
        p: int, order of the AR model;
    return:
        coeff: np.ndarray, AR coefficients;
        En: np.ndarray, final prediction error.

    Ref:M. Morf, etal, Recursive Multichannel Maximum Entropy Spectral Estimation,
                IEEE trans. GeoSci. Elec., 1978, Vol.GE-16, No.2, pp85-94.
        S. Haykin, Nonlinear Methods of Spectral Analysis, 2nd Ed.
            Springer-Verlag, 1983, Chapter 2
        Jie Cui, Lei Xu, Steven L. Bressler, Mingzhou Ding, Hualou Liang, 
            BSMART: a Matlab/C toolbox for analysis of multichannel neural time series, Neural Networks, 21:1094 - 1104, 2008.
    '''
    if x.ndim == 1: x = x[np.newaxis, :]
    
    L, N = x.shape  
    R0 = np.zeros((L, L)) 
    pf, pb, pfb, ap, bp, En = R0.copy(), R0.copy(), R0.copy(), R0.copy(), R0.copy(), R0.copy()

    for i in np.arange(1, Nr+1):
        En = En + x[:, (i-1)*Nl:i*Nl] @ x[:, (i-1)*Nl:i*Nl].T  
        ap = ap + x[:, (i-1)*Nl + 1:i*Nl] @ x[:, (i-1)*Nl + 1:i*Nl].T  
        bp = bp + x[:, (i-1)*Nl:i*Nl-1] @ x[:, (i-1)*Nl:i*Nl-1].T  

    ap = inv(cholesky(ap* (Nl-1)))  
    bp = inv(cholesky(bp* (Nl-1)))  

    for i in np.arange(1, Nr+1):
        
        efp = ap @ x[:, (i-1)*Nl+1:i*Nl] 
        ebp = bp @ x[:, (i-1)*Nl:i*Nl-1] 
        pf = pf + efp @ efp.T 
        pb = pb + ebp @ ebp.T  
        pfb = pfb + efp @ ebp.T  
    En = cholesky(En/N) 
    
    # Initial output variables
    coeff = []  
    kr=[]  

    for m in np.arange(1, p+1):
        ck = inv(cholesky(pf)) @ pfb @ inv(cholesky(pb).T)  
        kr.append(ck)  

        # Update the forward and backward prediction errors
        ef = np.eye(L) - ck @ ck.T  
        eb = np.eye(L) - ck.T @ ck  

        # Update the prediction error
        En = En @ cholesky(ef)  
        E = (ef +eb)/2  

        # Update the coefficients of the forward and backward prediction errors
        ap = np.dstack((ap, np.zeros((L, L))))  
        bp = np.dstack((bp, np.zeros((L, L))))  

        pf = np.zeros((L, L)) 
        pb = np.zeros((L, L))
        pfb = np.zeros((L, L))

        a = np.zeros((L, L, m + 1))
        b = np.zeros((L, L, m + 1))
        for i in np.arange(1, m+2):
            a[:, :, i-1] = inv(cholesky(ef)) @ (ap[:,:,i-1] - ck @ bp[:, :, m+1-i])
            b[:, :, i-1] = inv(cholesky(eb)) @ (bp[:,:,i-1] - ck.T @ ap[:, :, m+1-i])
        for k in np.arange(1, Nr+1):
            efp = np.zeros((L, Nl - m - 1))
            ebp = np.zeros((L, Nl - m - 1))

            for i in np.arange(1, m+2):
                k1 = m+2-i+(k-1)*Nl+1  
                k2 = Nl-i+1+(k-1)*Nl

                efp = efp + a[:, :, i-1] @ x[:, k1-1:k2]  
                ebp = ebp + b[:, :, m+1-i] @ x[:, k1-2:k2-1]  

            pf = pf + efp @ efp.T  
            pb = pb + ebp @ ebp.T  
            pfb = pfb + efp @ ebp.T  

        ap = a
        bp = b

    for j in np.arange(1, p+1):
        coeff.append(inv(a[:, :, 0]) @ a[:, :, j])  

    return -np.asarray(coeff), En @ En.T
