# Calculating iHmm Normal distribution Sample Gibbs States with Posterior
# Amir H. Poorjam

import numpy as np
import scipy.io
import scipy.special
import copy

def array2vector(array):
    array = array.reshape((array.shape[0], 1))
    return array

#######################
def iHmmHyperSample(Z, ibeta, ialpha0, igamma):
    K = ibeta.shape[1]-1        # number of states in iHmm
    tot_samples = Z.shape[1]          # length of iHmm
    # Compute N: state transition counts.
    N = np.zeros((K,K),dtype=np.int)
    N[0,int(Z[0,0])-1] = 1
    for t in range(1,tot_samples):
        N[int(Z[0,t-1])-1, int(Z[0,t])-1] = N[int(Z[0,t-1]-1), int(Z[0,t])-1] + 1
    
    # Compute M: number of similar dishes in each restaurant.
    M = np.zeros((K,K),dtype=np.int)
    for j in range(K):
        for k in range(K):
            if N[j,k] == 0:
                M[j,k] = 0
            else:
                for l in range(N[j,k]):
                    M[j,k] = M[j,k] + (np.random.uniform() < (ialpha0 * ibeta[0,k]) / (ialpha0 * ibeta[0,k] + l))
                                        
    # Resample beta
    alpha = array2vector(np.append(np.sum(M,axis=0),igamma))
    ibeta = dirichlet_sample(alpha)
    return ibeta, M, N

def dirichlet_sample(alpha):
    # Samples a dirichlet distributed random vector.
    dirichlet = np.random.gamma(alpha, scale=1)
    dirichlet = dirichlet / np.sum(dirichlet)
    return dirichlet.T

def NegLogLikelihood(fNewCluster, fMoS,fMoG,fMoGH,Yn, a0, b0, mu0, c0, mu_k, tau_k):
    N,D = Yn.shape
    mu0_repeated = np.tile(mu0,(N,1))
    b0_repeated  = np.tile(b0,(N,1))
    if fNewCluster:
        if fMoG:
            tau0 = (a0-0.5)/b0
            tau0_repeated = np.tile(tau0,(N,1))            
            negllk = D/2 * np.log(2*np.pi) + 0.5 * tau0_repeated * np.sum((Yn - mu0_repeated)**2, axis=1) - D/2 * np.log(tau0_repeated)
        
        elif fMoGH or fMoS:    
            if b0.size == 1:
                b0 = np.ones(1,D)*b0  # turn it to a row vector
                                    
            negllk = D/2 * np.log(2*np.pi) - D/2 * np.log(c0/(c0+1)) + D * scipy.special.gammaln(a0) - D * scipy.special.gammaln(a0 + 0.5) + 0.5 * np.sum(np.log(b0)) + (a0+0.5) * np.sum( np.log( 1 + (Yn - mu0_repeated)**2 * a0 * c0/(b0_repeated*(c0+1))/(2*a0) ) , axis=1)
    else:
        if fMoG or fMoGH:
            tau_k_repeated = np.tile(tau_k,(N,1))
            mu_k_repeated = np.tile(mu_k,(N,1))
            negllk = D/2 * np.log(2*np.pi) + np.sum(0.5 * tau_k_repeated * (Yn - mu_k_repeated)**2 - 0.5 * np.log(tau_k_repeated) , axis=1)
        
        elif fMoS: 
            negllk = D/2 * np.log(2*np.pi) - D/2 * np.log(c0/(c0+1))+ D*scipy.special.gammaln(a0) - D*scipy.special.gammaln(a0 + 0.5) + 0.5 * np.sum(np.log(b0)) + (a0+0.5) * np.sum( np.log( 1 + (Yn - mu0_repeated)**2 * a0 * c0/(b0_repeated*(c0+1))/(2*a0) ) , axis=1)
    
    return negllk

#######################
def main_iHMM_function(data, hypers, iterations, random_init_states):
    # Initialize the sampler.
    total_samples,D = data.shape
    sample = {'Z':random_init_states,
              'K':int(np.max(random_init_states))}
    
    # Setup dictionaries to store the output

    stats = {'K'      : np.zeros((1,iterations)),
             'alpha0' : np.zeros((1,iterations)),
             'gamma'  : np.zeros((1,iterations)),
             'jll'    : np.zeros((1,iterations))}
    
    # Initialize hypers; resample a few times as our inital guess might be off
    sample['alpha0'] = hypers['alpha0']
    sample['gamma']  = hypers['gamma']
    sample['Beta']   = np.ones((1, sample['K']+1)) / (sample['K']+1)
    sample['Beta'],_,_   = iHmmHyperSample(sample['Z'], sample['Beta'], sample['alpha0'], sample['gamma'])
    
#    samplebeta = scipy.io.loadmat('sample_beta.mat') # for debugging
#    sample['Beta'] = samplebeta['sampleBeta']        # for debugging
    
    ittr = 0 
    posterior = np.zeros((1,total_samples))
    N_list = []
    M_list = []
    SumObserv_list = []
    SumSquaredObserv_list = []
    NumberObserv_list = []
    
    while ittr < iterations:
        print('Iteration: ' + str(ittr+1), end =" ")
        # Compute the sufficient statistics for the normal distribution       
        SumObserv        = np.zeros((sample['K'], D))
        SumSquaredObserv = np.zeros((sample['K'], D))
        NumberObserv     = np.zeros((sample['K'], 1),dtype=np.int)
        
        for t in range(total_samples):
            SumObserv[int(sample['Z'][0,t])-1, :] = SumObserv[int(sample['Z'][0,t])-1, :] + data[t, :]
            SumSquaredObserv[int(sample['Z'][0,t])-1, :] = SumSquaredObserv[int(sample['Z'][0,t])-1, :] + data[t, :]**2
            NumberObserv[int(sample['Z'][0,t])-1,0] =  NumberObserv[int(sample['Z'][0,t])-1,0] + 1
            
        
        # Compute the empirical transition matrix. N(i,j) is the number of transition from state i to j
        N = np.zeros((sample['K'], sample['K']),dtype=np.int)
        N[0, int(sample['Z'][0,0])-1] = 1
        for t in range (1,total_samples):
            N[int(sample['Z'][0,t-1])-1, int(sample['Z'][0,t])-1] = N[int(sample['Z'][0,t-1])-1, int(sample['Z'][0,t])-1] + 1

        # Start resampling the hidden state sequence.
        for t in range(total_samples):
            # Discount the transition and emission counts for timestep t
            SumObserv[int(sample['Z'][0,t])-1, :] = SumObserv[int(sample['Z'][0,t])-1, :] - data[t, :]
            SumSquaredObserv[int(sample['Z'][0,t])-1, :] = SumSquaredObserv[int(sample['Z'][0,t])-1, :] - data[t, :]**2
            NumberObserv[int(sample['Z'][0,t])-1,0] = NumberObserv[int(sample['Z'][0,t])-1,0] - 1
            
            if t != 0:
                N[int(sample['Z'][0,t-1])-1, int(sample['Z'][0,t])-1] = N[int(sample['Z'][0,t-1])-1, int(sample['Z'][0,t])-1] - 1
            else:
                N[0, int(sample['Z'][0,t])-1] = N[0, int(sample['Z'][0,t])-1] - 1
            
            if t != total_samples-1:
                N[int(sample['Z'][0,t])-1, int(sample['Z'][0,t+1])-1] = N[int(sample['Z'][0,t])-1, int(sample['Z'][0,t+1])-1] - 1
                
            
            # Compute the marginal probability for timestep t
            r = np.ones((1, sample['K']+1))
            for k in range(sample['K']):
                if t != 0:
                    r[0,k] = r[0,k] * ( N[int(sample['Z'][0,t-1])-1, k] + sample['alpha0'] * sample['Beta'][0,k] )
                else:
                    r[0,k] = r[0,k] * ( N[0, k] + sample['alpha0'] * sample['Beta'][0,k] )
                
                
                if t != total_samples-1:
                    if t > 0 and k != int(sample['Z'][0,t-1])-1:
                        r[0,k] = r[0,k] * ( N[k, int(sample['Z'][0,t+1])-1] + sample['alpha0'] * sample['Beta'][0,int(sample['Z'][0,t+1])-1] ) / ( sum(N[k, :]) + sample['alpha0'] )
                        
                    elif t == 0 and k != 0:
                        r[0,k] = r[0,k] * ( N[k, int(sample['Z'][0,t+1])-1] + sample['alpha0'] * sample['Beta'][0,int(sample['Z'][0,t+1])-1] ) / ( sum(N[k, :]) + sample['alpha0'] )
                    
                    elif t > 0 and k == int(sample['Z'][0,t-1])-1 and k != int(sample['Z'][0,t+1])-1:
                        r[0,k] = r[0,k] * ( N[k, int(sample['Z'][0,t+1])-1] + sample['alpha0'] * sample['Beta'][0,int(sample['Z'][0,t+1])-1] ) / ( sum(N[k, :]) + 1 + sample['alpha0'] )
                    
                    elif t > 0 and k == int(sample['Z'][0,t-1])-1 and k == int(sample['Z'][0,t+1])-1:
                        r[0,k] = r[0,k] * ( N[k, int(sample['Z'][0,t+1])-1] + 1 + sample['alpha0'] * sample['Beta'][0,int(sample['Z'][0,t+1])-1] ) / ( sum(N[k, :]) + 1 + sample['alpha0'] )
                    
                    elif t == 0 and k == 0 and k != int(sample['Z'][0,t+1])-1:
                        r[0,k] = r[0,k] * ( N[k, int(sample['Z'][0,t+1])-1] + sample['alpha0'] * sample['Beta'][0,int(sample['Z'][0,t+1])-1] ) / ( sum(N[k, :]) + 1 + sample['alpha0'] )
                    
                    elif t == 0 and k == 0 and k == int(sample['Z'][0,t+1])-1:
                        r[0,k] = r[0,k] * ( N[k, int(sample['Z'][0,t+1])-1] + 1 + sample['alpha0'] * sample['Beta'][0,int(sample['Z'][0,t+1])-1] ) / ( sum(N[k, :]) + 1 + sample['alpha0'] )
                        
                
                # Update Cluster parameters
                a_n = hypers['a0'] + NumberObserv[k, 0]/2
                c_n = hypers['c0'] + NumberObserv[k, 0]
                m_n = ((hypers['c0']*hypers['m0'] + array2vector(SumObserv[k, :]))/(hypers['c0'] + NumberObserv[k,0])).T
                if NumberObserv[k, 0] != 0:
                    b_n = hypers['b0'] + (array2vector(0.5*(SumSquaredObserv[k, :] - (SumObserv[k, :]**2)/NumberObserv[k, 0])).T) + (((hypers['c0']*NumberObserv[k, 0]*(array2vector(SumObserv[k, :]/NumberObserv[k, 0]) - hypers['m0'])**2)/(2*(hypers['c0'] + NumberObserv[k, 0]))).T)
                else:
                    b_n = hypers['b0']
                    
                r[0,k] = r[0,k] * np.exp(-NegLogLikelihood(0, 1, 0, 0, array2vector(data[t, :]).T, a_n, b_n, m_n, c_n,[],[]))
                    
                    
                
            r[0,sample['K']] = np.exp(-NegLogLikelihood(1, 1, 0, 1, array2vector(data[t, :]).T, hypers['a0'], hypers['b0'], hypers['m0'].T, hypers['c0'],[],[]))


            if t != total_samples-1:
                r[0,sample['K']] = r[0,sample['K']] * sample['Beta'][0,int(sample['Z'][0,t+1])-1]
        
        
            # Resample s_t
            r = r / np.sum(r)
            sample['Z'][0,t] = 1 + np.sum(np.random.uniform() > np.cumsum(r))

#            sampleZ = scipy.io.loadmat('sampleZ.mat') # for debugging
#            sample['Z'] = sampleZ['sampleZ']          # for debugging


            # If we move to a new state
            if sample['Z'][0,t] > sample['K']:
                zero_vector = np.full((N.shape[0]+1,N.shape[0]+1),0)
                zero_vector[0:N.shape[0],0:N.shape[1]]=N
                N = copy.deepcopy(zero_vector)
                
                SumObserv = np.concatenate((SumObserv,np.zeros((1,SumObserv.shape[1]))),axis=0)
                SumSquaredObserv = np.concatenate((SumSquaredObserv,np.zeros((1,SumSquaredObserv.shape[1]))),axis=0)
                NumberObserv = np.concatenate((NumberObserv,np.zeros((1,1))))

                
                # Extend Beta using standard stick-breaking construction
                b = np.random.beta(1, sample['gamma'])
                BetaU = sample['Beta'][0,-1]
                sample['Beta'][0,-1] = b * BetaU
                sample['Beta'] = array2vector(np.append(sample['Beta'],(1-b)*BetaU)).T
                sample['K'] = sample['K'] + 1
                
            # Update emission statistics and transition counts.
            SumObserv[int(sample['Z'][0,t])-1, :] = SumObserv[int(sample['Z'][0,t])-1, :] + data[t, :]
            SumSquaredObserv[int(sample['Z'][0,t])-1, :] = SumSquaredObserv[int(sample['Z'][0,t])-1, :] + data[t, :]**2
            NumberObserv[int(sample['Z'][0,t])-1,0] =  NumberObserv[int(sample['Z'][0,t])-1,0] + 1
        

            if t != 0:
                N[int(sample['Z'][0,t-1])-1, int(sample['Z'][0,t])-1] = N[int(sample['Z'][0,t-1])-1, int(sample['Z'][0,t])-1] + 1
            else:
                N[0, int(sample['Z'][0,t])-1] = N[0, int(sample['Z'][0,t])-1] + 1
            
            if t != total_samples-1:
                N[int(sample['Z'][0,t])-1, int(sample['Z'][0,t+1])-1] = N[int(sample['Z'][0,t])-1, int(sample['Z'][0,t+1])-1] + 1
            
        
        # Recompute the number of states - recycle empty ones.
        zind = (np.sort(np.setdiff1d(np.arange(1,sample['K']+1), sample['Z'])))[::-1]
        if zind.size != 0: # we have new states
            for zix in zind:             # We sorted decending to make sure we delete from the back onwards, otherwise indexing is more complex
                sample['Z'][sample['Z'] > zix] = sample['Z'][sample['Z'] > zix] - 1
                sample['Beta'][0,-1] = sample['Beta'][0,-1] + sample['Beta'][0,zix-1]
                sample['Beta'] = array2vector(np.delete(sample['Beta'], zix-1)).T
    
        sample['K'] = sample['Beta'].shape[1]-1
        
        # Resample Beta
        sample['Beta'],M,N  = iHmmHyperSample(sample['Z'], sample['Beta'], sample['alpha0'], sample['gamma'])
        
        # Prepare next iteration.
        SO, SSO, NO = SumObserv, SumSquaredObserv, NumberObserv
        
        SO  = np.delete(SO, zind, axis=0)
        SSO = np.delete(SSO, zind, axis=0)
        NO  = np.delete(NO, zind)
                
        N_list.append(N)
        M_list.append(M)
        SumObserv_list.append(SO)
        SumSquaredObserv_list.append(SSO)
        NumberObserv_list.append(NO)
        
        stats['alpha0'][0,ittr] = sample['alpha0']
        stats['gamma'][0,ittr] = sample['gamma']
        stats['K'][0,ittr] = sample['K']
        
        
        posterior = np.concatenate((posterior,(sample['Z'])))

        ittr += 1
        print(', K: ' + str(sample['K']))
    stats['N'] = N_list
    stats['M'] = M_list
    stats['SumObserv'] = SumObserv_list
    stats['SumSquaredObserv'] = SumSquaredObserv_list
    stats['NumberObserv'] = NumberObserv_list

#    mode_parameters = np.argmax(stats['jll'])
    mode_parameters = iterations-1
    suff_stats_pre = {}
    suff_stats_pre['K'] = stats['K'][0,mode_parameters]
    suff_stats_pre['N'] = N_list[mode_parameters]
    suff_stats_pre['M'] = M_list[mode_parameters]
    suff_stats_pre['NumberObserv'] = NumberObserv_list[mode_parameters]
    suff_stats_pre['SumSquaredObserv'] = SumSquaredObserv_list[mode_parameters]
    suff_stats_pre['SumObserv'] = SumObserv_list[mode_parameters]
    suff_stats_pre['mode_parameters'] = mode_parameters
    
    return posterior, stats, suff_stats_pre

    