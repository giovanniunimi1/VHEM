from ast import Pass
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import time
import gmm
def vbhmm_em(data, K, ini):
        VERBOSE_MODE = ini["verbose"]

        # get length of each chain
        trial = len(data)
        datalen = [len(d) for d in data]
        lengthT = max(datalen)
        totalT = sum(datalen)
       # print(data[1])
        #print(datalen.shape)

        # initialize the parameters
        mix_t = vbhmm_init(data, K, ini)  # initialize the parameters
        mix = mix_t
        dim = mix["dim"]
        K = mix["K"]
        N = trial
        maxT = lengthT
        alpha0 = mix["alpha0"]
        epsilon0 = mix["epsilon0"]
        m0 = mix["m0"]
        beta0 = mix["beta0"]
        v0 = mix["v0"]
        W0inv = mix["W0inv"]
        alpha = mix["alpha"]
        epsilon =np.array(mix["epsilon"])
        beta = mix["beta"]
        v = mix["v"]
        m = mix["m"]
        W = mix["W"]
        C = mix["C"] 
        const = mix["const"]
        const_denominator = mix["const_denominator"]
        maxIter = ini["maxIter"]
        minDiff = ini["minDiff"]  
        L = -float("inf")
        lastL = -float("inf")
        logLambdaTilde = np.zeros(K)
        psiEpsilonHat = np.zeros(K)
        psiEpsilon=np.zeros(K)
        logATilde = np.zeros((K, K))
       
        #else:
            #usegroups = 0
        #INIZIO ALGORITMO
        for iter in range(1,maxIter+1):
            print(iter)

            #pre-calculate constants logL
            for k in range(K):

                t1 = sp.psi(0.5 *np.tile((v[k] + 1), (dim, 1))) - sp.psi(0.5 * np.arange(1, dim + 1))
                #10.65
                logLambdaTilde[k] = np.sum(t1 ) + const + np.log(np.linalg.det(np.abs(W[:,:,k])))
                #non presente, sfruttiamo epsilon
                #print(epsilon[:,k])
                psiEpsilonHat[k]=sp.psi( np.sum(epsilon[:, k]))
                psiEpsilon = sp.psi(epsilon[:,k])
                logATilde[:,k] = psiEpsilon - psiEpsilonHat[k]
             #10.66
            logPiTilde=sp.psi(alpha)-sp.psi(sum(alpha))

            logrho_Saved = np.zeros((K, N, maxT))
            fb_qnorm = np.zeros(N)
            gamma_sum = np.zeros((K, N, maxT))
            sumxi_sum = np.zeros((K, K, N))
            t_gamma_Saved = []
            for n in range(N):
                tdata = data[n]
                tdata = np.transpose(tdata)
                #print(tdata)
                tT = tdata.shape[1]
            
                #print(tdata.shape)
                #print(tT)
                delta = np.zeros((K, tT))
                logrho = np.zeros((K, tT))
                for k in range(K):
                    diff = tdata - m[:, k][:, np.newaxis]
                    delta[k, :] = dim / beta[k] + v[k] * np.sum((W[:, :, k] @ diff) * diff, axis=0)
                #10.46 modificata
                logrho = 0.5 * np.subtract(logLambdaTilde[:, np.newaxis], delta) - const_denominator
                logrho_Saved[:, n, :tT] = logrho
                # FORWARD BACKWARD PHASE

                gamma = np.zeros((K, tT))
                sumxi = np.zeros((K, K)) #[row]
                t_alpha=np.zeros((tT, K))
                t_beta=[]
                t_c=np.zeros(tT)

                t_logPiTilde=np.exp(logPiTilde.T) #prior ##
                t_logATilde=np.exp(logATilde.T) #transition
                t_logrho=np.exp(logrho.T)  #emission
                t_x=np.transpose(tdata)
                t_T = t_x.shape[0]
                if t_T >= 1:
                    # forward!!!
                    t_gamma = []
                    t_sumxi = np.zeros((K, K)) 
                    
                    t_alpha[0, :] = t_logPiTilde * t_logrho[0, :] 

                    # Rescaling for numerical stability
                    t_c[0] = np.sum(t_alpha[0, :])
                    t_alpha[0, :] /= t_c[0]
                if t_T > 1:

                    for i in range(1, t_T):
                        #print('alpha')
                        #print(t_alpha[i-1, :])
                        #print('logatilde')
                        #print(t_logATilde)
                        #print(t_logATilde.shape)
                        #print(t_logrho[i, :])
                        
                        t_alpha[i, :] = t_alpha[i-1, :]@t_logATilde*t_logrho[i, :]

                        # Rescaling for numerical stability
                        t_c[i] = np.sum(t_alpha[i, :])
                        t_alpha[i, :] /= t_c[i]

                t_beta = np.zeros((t_T, K))
                t_gamma = np.zeros((K, t_T))
                t_beta[t_T-1, :] = np.ones(K) / K
                t_gamma[:, t_T-1] = t_alpha[t_T-1, :] * t_beta[t_T-1, :].T
                #backward
                if t_T > 1:
                    for i in range(t_T-2, -1, -1):
                        bpi = t_beta[i+1, :] * t_logrho[i+1, :]
                        t_beta[i, :] = np.dot(bpi, t_logATilde.T)
                        t_beta[i, :] /= t_c[i+1]
                        t_gamma[:, i] = t_alpha[i, :] * t_beta[i, :].T
        
                        tmp_xi = t_logATilde * (np.outer(t_alpha[i, :], bpi) / t_c[i+1])
                        tmp_xi /= np.sum(tmp_xi)
                        t_sumxi += tmp_xi
    
                for i in range(t_gamma.shape[1]):
                    gamma[:, i] = t_gamma[:, i]
                sumxi = t_sumxi

                gamma_sum[:, n, 0:tT] = gamma
                sumxi_sum[:, :, n] = sumxi

                fb_qnorm[n] = np.sum(np.log(t_c))
            scale = np.sum(gamma_sum, axis=0)
            scale[scale == 0] = 1
            #print('Alpha')
            #print(t_alpha.shape)
            #print('Beta')
            #print(t_beta.shape) 
            #print('Gamma')
            #print(t_gamma.shape)
            t_gamma_Saved = np.divide(gamma_sum, scale)  # marginal responsibilities
            #print('Gamma saved shape')
            #print(t_gamma_Saved.shape)
            t_Nk1 = np.reshape(np.sum(t_gamma_Saved, axis=1), (K, maxT))
            Nk1 = t_Nk1[:, 0]
            Nk1 = Nk1 + 1e-50
            Nk = np.sum(t_Nk1, axis=1)
            Nk = Nk + 1e-50
            M = np.sum(sumxi_sum, axis= 2)
            t_xbar = np.zeros((K, N, maxT, dim))
            for n in range(N):
                x = data[n]
                #print(x)
                x = np.array(x)
                tT = x.shape[0]
                #print(tT)
                t_xbar[:, n, 0:tT, :] = np.multiply(
                   t_gamma_Saved[:, n, 0:tT, np.newaxis],
                   x[np.newaxis, np.newaxis, 0:tT, :]
                )
            t1_xbar = np.sum(t_xbar, axis=2)
            t2_xbar = np.sum(t1_xbar, axis=1)
            t2_xbar = np.reshape(t2_xbar, (K, dim))

            xbar = np.divide(t2_xbar, Nk[:, np.newaxis])

            t1_S = np.zeros((dim, dim, K))
            for n in range(N):
                x = np.transpose(data[n])
                tT = x.shape[1]
                for k in range(K):
                    d1 = x - np.reshape(xbar[k, :], (dim, 1))
                    d2 = np.multiply(np.reshape(t_gamma_Saved[k, n, 0:tT], (1, tT)), d1)
                    t1_S[:, :, k] += np.dot(d1, d2.T)
            t1_S = np.divide(t1_S, np.reshape(Nk, (1, 1, K)))

            # Calculate the lower bound
            # ABC (11)

            # Constants
            logCalpha0 = sp.gammaln(K*alpha0) - K*sp.gammaln(alpha0)
            logCepsilon0 = np.zeros(K)
            for k in range(K):
                logCepsilon0[k] = sp.gammaln(K*epsilon0) - K*sp.gammaln(epsilon0)
            logB0 = (v0/2)*np.log(np.linalg.det(W0inv)) - (v0*dim/2)*np.log(2) \
                - (dim*(dim-1)/4)*np.log(np.pi) - np.sum(sp.gammaln(0.5*(v0+1 -np.arange(1,dim+1))))
            logCalpha = sp.gammaln(np.sum(alpha)) - np.sum(sp.gammaln(alpha))
            logCepsilon = np.zeros(K)
            for k in range(K):
                logCepsilon[k] = sp.gammaln(np.sum(epsilon[:,k])) - np.sum(sp.gammaln(epsilon[:,k]))
            H = 0
            #dichiarare trSW xbarWxbar e trW0invW
            trSW = np.zeros(K) 
            xbarWxbar = np.zeros(K)
            mWm = np.zeros(K)
            trW0invW = np.zeros(K)
            for k in range(K):
              logBk = -(v[k]/2)*np.log(np.linalg.det(W[:,:,k])) - (v[k]*dim/2)*np.log(2) \
                - (dim*(dim-1)/4)*np.log(np.pi) - np.sum(sp.gammaln(0.5*(v[k] + 1 - np.arange(1,dim+1))))
              H = H - logBk - 0.5*(v[k] - dim - 1)*logLambdaTilde[k] + 0.5*v[k]*dim
              trSW[k] = np.trace(v[k]*np.dot(t1_S[:,:,k],W[:,:,k]))
              xbarT = xbar[k,:].T
              diff = xbarT - m[:,k]
              xbarWxbar[k] = diff.T @ W[:,:,k] @ diff
              diff = m[:,k] - m0
              mWm[k] = diff.T @ W[:,:,k] @ diff
              trW0invW[k] = np.trace(W0inv @ W[:,:,k])

            # E(log p(X|Z,mu,Lambda)  Bishop (10.71) - ABC term 1
            Lt1 = 0.5*np.sum(Nk*(logLambdaTilde - dim/beta \
              - trSW - v*xbarWxbar - dim*np.log(2*np.pi)))

            # Initial responsibilities (t=1)
            gamma1 = t_gamma_Saved[:,:,0]

            # E[log p(Z|pi)]   Bishop (10.72) - ABC term 2, part 1
            PiTilde_t = np.tile(logPiTilde,(1,N))
            gamma_t1 = gamma1*PiTilde_t
            Lt2a = np.sum(gamma_t1)

            # E[log p(Z|A)]   ~Bishop 10.72 - ABC term 2, part 2 [CORRECT?]
            
            ATilde_t = logATilde.T
            Lt2b = np.sum(M*ATilde_t)
            

            # E[log p(Z|pi, A)]
            # ABC term 2
            Lt2 = Lt2a + Lt2b

            # E[log p(pi)]   Bishop (10.73)   ABC term 3
            Lt3 = logCalpha0 + (alpha0-1)*np.sum(logPiTilde)

            # E[log p(A)] = sum E[log p(a_j)]   (equivalent to Bishop 10.73) ABC term 4
            Lt4a = logCepsilon0 + (epsilon0 - 1)*np.sum(logATilde) ##not sure
            Lt4 = np.sum(Lt4a)
            # E[log p(mu, Lambda)]  Bishop (10.74)  ABC term 5
            Lt51 = 0.5*np.sum(dim*np.log(beta0/(2*np.pi)) + logLambdaTilde - dim*beta0/beta - beta0*v*mWm)
            Lt52 = K*logB0 + 0.5*(v0-dim-1)*np.sum(logLambdaTilde) - 0.5*np.sum(v*trW0invW)
            Lt5 = Lt51 + Lt52
            # 2016-04-26 ABC: use correct q(Z)

            # 2016-04-29 ABC: E[z log pi] (same as Lt2a)
            # Lt61 = np.sum(np.multiply(t_gamma_Saved[:,:,0], logPiTilde))
            Lt61 = Lt2a

            # 2016-04-29 ABC:  E[zt zt-1 log a]  (same as Lt2b)
            Lt62 = Lt2b

            # 2016-04-29 ABC:  E[z log rho]
            # The zeros in logrho should remove times (tT+1):N
            Lt63 = np.sum(np.sum(np.sum(np.multiply(t_gamma_Saved, logrho_Saved))))

            # 2016-04-29 ABC: normalization constant for q(Z)
            Lt64 = np.sum(fb_qnorm)
            # print('   norm constant:', Lt64)

            # 2016-04-29 ABC: E[log q(Z)] - ABC term 6
            Lt6 = Lt61 + Lt62 + Lt63 - Lt64

            # E[log q(pi)]  Bishop (10.76)

            Lt71 = np.sum((alpha - 1)*logPiTilde) + logCalpha

            # E[log q(aj)]  (equivalent to Bishop 10.76)

            Lt72 = np.sum((epsilon - 1)*logATilde) + logCepsilon
            Lt72sum = np.sum(Lt72)

            # E[log q(pi, A)] - ABC term 7
            Lt7 = Lt71 + Lt72sum

            # E[q(mu,Lamba)]  Bishop (10.77) - ABC term 8
            Lt8 = 0.5*np.sum(logLambdaTilde + dim*np.log(beta/(2*np.pi))) - 0.5*dim*K - H

            if iter > 1:
              lastL = L
            L = Lt1 + Lt2 + Lt3 + Lt4 + Lt5 - Lt6 - Lt7 - Lt8
            #M-STEP
            
            alpha = alpha0 + Nk1
            alpha = alpha.reshape((3,1))
            epsilon = epsilon0 + M
            
            epsilon = epsilon.T
            
            # Update Gaussians
            #if not ini.fix_clusters:
              # 
            
            beta = beta0 + Nk
            v = v0 + Nk + 1
            
            for k in range(K):
              m[:,k] = (beta0*m0 + Nk[k]*xbar[k,:]) / beta[k]

              # Wishart
              for k in range(K):
                #if ini.fix_cov is None:
                mult1 = beta0*Nk[k] / (beta0 + Nk[k])
                diff3 = xbar[k,:] - np.array(m0).T
                diff3 = diff3.T
                W[:,:,k] = np.linalg.inv(W0inv + Nk[k]*t1_S[:,:,k] + mult1*np.outer(diff3, diff3))
                

            # Covariance
            for k in range(K):
              C[:,:,k] = np.linalg.inv(W[:,:,k]) / (v[k]-dim-1)

            if iter > 1:
              likIncr = np.abs((L-lastL)/lastL)
              if VERBOSE_MODE >= 3:
                print(iter, ': L=', L, '; dL=', likIncr)
                if L-lastL < 0:
                  print(' !!!')
                  # keyboard
                print()
              else:
                if L-lastL < 0:
                  if VERBOSE_MODE >= 2:
                    print('LL decreased')
              if likIncr <= minDiff:
                break
        # Generate the output model
        # NOTE: if adding a new field, remember to modify vbhmm_permute.
        hmm = {}
        prior_s = np.sum(alpha)
        prior = alpha / prior_s
        hmm['prior'] = prior
        trans_t = epsilon.T  # [row]
        for k in range(K):
            scale = np.sum(trans_t[k,:])
            if scale == 0:
                scale = 1
            trans_t[k,:] = trans_t[k,:] / scale
        hmm['trans'] = trans_t
        hmm['pdf'] = []
        for k in range(K):
            pdf = {}
            pdf['mean'] = m[:,k].T
            pdf['cov'] = C[:,:,k]
            hmm['pdf'].append(pdf)

        hmm['LL'] = L
        hmm['gamma'] = [] 
        for n in range(N):
            hmm['gamma'].append(np.reshape(t_gamma_Saved[:, n, :datalen[n]], (K, datalen[n])))
        hmm['M']=M 
        hmm['N1']=Nk1 
        hmm['N']=Nk 
        hmm['varpar'] = {}
        hmm['varpar']['epsilon']= epsilon
        hmm['varpar']['alpha']= alpha
        hmm['varpar']['beta']= beta
        hmm['varpar']['v']= v
        hmm['varpar']['m']= m
        hmm['varpar']['W']=W
        return hmm

###Corretta (init)
def vbhmm_init(datai,K,ini):
        #Inizializza le variabili, richiama vbhmm_gmm per inizializzare le probabilità di emissione
        VERBOSE_MODE=ini['verbose']

        data = np.concatenate(datai, axis=0)
        N, dim = data.shape
        #initmode
        if ini['initmode']=='random':
            try:
                    mix = GaussianMixture(n_components=K, covariance_type='diag', reg_covar=0.0001, tol=1e-5)
                    mix.fit(data)
            except Exception as e:
                if 'IllCondCovIter' in str(e):
                    if VERBOSE_MODE >= 2:
                        print('using shared covariance')
                    mix=GaussianMixture(n_components=K, covariance_type='diag', shared_covariance=True, tol=1e-5)
                    mix.fit(data)
                else:
                    if VERBOSE_MODE >= 2:
                        print('using built-in GMM')
                    gmmmix = gmm.gmm_learn(data.T, K, {'cvmode': 'full', 'initmode': 'random', 'verbose': 0})
                    mix.weights_ = np.array(gmmmix['pi'])
                    mix.means_ = np.concatenate(gmmmix['mu'], axis=1).T
                    mix.covariances_ = np.array(gmmmix['cv'])
        elif ini['initmode']=='initgmm':

            mix = GaussianMixture(n_components=K)
            mix.covariances_ = np.array(ini['initgmm']['cov'])
            mix.means_ = np.concatenate(ini['initgmm']['mean'], axis=0)
            if mix.means_.shape[0] != K:
                raise ValueError('bad initgmm dimensions -- possibly mean is not a row vector')
            mix.weights_ = np.array(ini['initgmm']['prior'])
        elif ini['initmode']=='split':
            gmmmix = gmm.gmm_learn(data.T, K, {'cvmode': 'full', 'initmode': 'split', 'verbose': 0})
            mix.weights_ = np.array(gmmmix['pi'])
            mix.means_ = np.concatenate(gmmmix['mu'], axis=1).T
            mix.covariances_ = np.array(gmmmix['cv'])
        else:
            print('Error,bad init mode')
    
        mix_t = {}
        mix_t['dim'] = dim
        mix_t['K'] = K
        mix_t['N'] = N
        #print(N)
        # setup hyperparameters
        mix_t['alpha0'] = ini['alpha']
        mix_t['epsilon0'] = ini['epsilon']
        if len(ini['mu']) != dim:
            raise ValueError(f'vbopt.mu should have dimension D={dim}')
        mix_t['m0'] = ini['mu']
        mix_t['beta0'] = ini['beta']
        if isinstance(ini['W'], float):     
            # isotropic W
            mix_t['W0'] = ini['W'] * np.eye(dim)  # 2016/04/26: ABC BUG Fix: was inv(ini.W*eye(dim))
        else:
            # diagonal W
            if len(ini['W']) != dim:
                raise ValueError(f'vbopt.W should have dimension D={dim} for diagonal matrix')
            mix_t['W0'] = np.diag(ini['W'])
        if ini['v'] <= dim - 1:
            raise ValueError('v not large enough')
        mix_t['v0'] = ini['v']  # should be larger than p-1 degrees of freedom (or dimensions)
        mix_t['W0inv'] = np.linalg.inv(mix_t['W0'])
        mix_t['epsilon']=np.zeros((K, K))
        # setup model (M-step)
        mix_t['Nk'] = N * mix.weights_
        mix_t['Nk2'] = (N/K) * np.ones((K, 1))
        #print(mix_t['Nk2'])
        mix_t['xbar'] = mix.means_.T
        #print(mix.covariance_type)
        #print(mix_t['xbar'])#
        if mix.covariance_type == 'full':
            mix_t['S'] = mix.covariances_
        elif mix.covariance_type == 'diag':
            mix_t['S'] = np.zeros((dim, dim, K))
            for j in range(K):
                mix_t['S'][:, :, j] = np.diag(mix.covariances_[j,:])
        
        # handle diagonal Sigma (for Antoine)
        if ((mix_t['S'].shape[0] == 1) or (mix_t['S'].shape[1] == 1)) and (dim > 1):
            oldS = mix_t['S']
            mix_t['S'] = np.zeros((dim, dim, K))
            for j in range(K):
                mix_t['S'][:,:,j] = np.diag(oldS[:,:,j])  # make the full covariance
        #print(mix_t['S'][:,:,0])#
        mix_t['alpha'] = mix_t['alpha0'] + mix_t['Nk2']
        for k in range(K):
            mix_t['epsilon'][:, k] = mix_t['epsilon0'] + mix_t['Nk2'].flatten()
        #print(mix_t['epsilon'])
        mix_t['beta'] = mix_t['beta0'] + mix_t['Nk']
        mix_t['v'] = mix_t['v0'] + mix_t['Nk'] + 1  # 2016/04/26: BUG FIX (add 1)g
        variabile = np.array(mix_t['beta0'] * mix_t['m0'])
        mix_t['m'] =(variabile[:, np.newaxis]) * np.ones((1, K)) + ((np.ones((dim, 1)) * mix_t['Nk'].T) * mix_t['xbar'])/(np.ones((dim, 1)) * mix_t['beta'].T)
        mix_t['W'] = np.zeros((dim, dim, K))
        for k in range(K):
            mult1 = mix_t['beta0'] * mix_t['Nk'][k] / (mix_t['beta0'] + mix_t['Nk'][k])
            diff3 = mix_t['xbar'][:,k] - mix_t['m0']
            #print(mix_t['W0inv'].shape)#
            #print(mix_t['Nk'][k].shape)#
            #print(mix_t['S'][:,:,k].shape)#
            #print(np.outer(diff3, diff3).shape)#
            outer_term = mult1 * np.outer(diff3, diff3)
            mix_t['W'][:, :, k] = np.linalg.inv(np.linalg.inv(mix_t['W0inv']) + mix_t['Nk'][k] * mix_t['S'][:, :, k] + np.diag(outer_term)) ##mult1 * np.outer(diff3, diff3) come un termine diagonale
        mix_t['C'] = np.zeros((dim, dim, K))
        mix_t['const'] = dim * np.log(2)
        mix_t['const_denominator'] = (dim * np.log(2 * np.pi)) / 2
        #for key, value in mix_t.items():
            #print(f"Key: {key}, Value: {value}")
        return mix_t




#######VBHMM_LEARN FUNCTION####### (COMPLETA E CORRETTA)
def learn(data, K, vbopt=None):
        
    
    ###INIZIALIZZAZIONE VARIABILI OPZIONALI
        if vbopt is None:
            vbopt = {}
        D = 2      
        defmu = [256, 192]
        vbopt = {
            'alpha':0.1,
            'mu': defmu,  # hyper-parameter for the mean
            'W': 0.005,  # the inverse of the variance of the dimensions
            'beta': 1,
            'v': 5,
            'epsilon': 0.1,
            'initmode': 'random',
            'numtrials': 20,
            'maxIter': 100,  # maximum allowed iterations
            'minDiff': 1e-5,
            'showplot': 1,
            'sortclusters': 'f',
            'groups': [],
            'fix_clusters': 0,
            'random_gmm_opt': {},
            'fix_cov': [],
            'verbose': 3
}
        VERBOSE_MODE=vbopt['verbose']
    ###PER PIU' VALORI DI K (DI SOLITO K=[1,2,3])
        if isinstance(K, list):
        #turn off plotting
            vbopt2 = vbopt;
            vbopt2['showplot']=0 
            out_all = []
            LLk_all = []
            for ki in range(len(K)):
                if VERBOSE_MODE >= 2:
                    print(f'-- K={K[ki]} --')
                elif VERBOSE_MODE == 1:
                    print(f'-- vbhmm K={K[ki]}: ', end='')
                if vbopt2['initmode'] == 'initgmm':
                    vbopt2['initgmm'] = vbopt.initgmm[ki]

                out_all.append(learn(data, K[ki], vbopt2))
                LLk_all.append(out_all[ki].LL)
            LLk_all = LLk_all + sp.gammaln(K+1)

            #get K with max data likelihood
            maxLLk = np.max(LLk_all)
            ind = np.argmax(LLk_all)

            # return the best model
            hmm = out_all[ind]
            hmm.model_LL = LLk_all
            hmm.model_k = K
            hmm.model_bestK = K[ind]
            L = maxLLk

            if VERBOSE_MODE >= 1:
                print(f"best model: K={K[ind]}; L={maxLLk}")


        else:
            if vbopt['initmode'] == 'random':
                numits=vbopt['numtrials']
                vb_hmms=[]
                LLall = np.zeros(numits)
                for it in range(numits):
                    if VERBOSE_MODE==1:
                        print(it+1, end=' ')
                    hmm = vbhmm_em(data, K, vbopt)
                    vb_hmms.append(hmm)
                    LLall[it] = hmm['LL']
                maxind = max(range(len(LLall)), key=lambda i: LLall[i])
                maxLL = LLall[maxind]
                if VERBOSE_MODE >= 1:
                    print(f'\nbest run={maxind}; LL={maxLL}')

                hmm = vb_hmms[maxind]
                L = hmm['LL']
                #print(hmm)
                #print(hmm['prior'])
                #print(hmm['prior'].shape)
            ###Initialize with learned GMM###
            elif vbopt['initmode'] == 'initgmm':
                hmm=vbhmm_em(data,K,vbopt)
                L=hmm.LL

            else:
                pass


        # append the options
        hmm['vbopt'] = vbopt
        return hmm