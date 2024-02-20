
from ast import Pass
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import time

#####VHEM_CLUSTER##### omega : pesi di mischiamento delle distribuzioni di probabilità
def vhem_cluster(hmms, K, S=None, hemopt=None):
        if S is None:
            S = np.median([len(hmm['prior']) for hmm in hmms])
            print(f'using median number of states: {S}')

        if hemopt is None:
            hemopt = {}

        hemopt.setdefault('K', K)
        hemopt.setdefault('verbose', 2)
        #VERBOSE_MODE = hemopt['verbose']
        #setable parameters (could be changed)
        hemopt.setdefault('N', S)
        hemopt.setdefault('trials', 50)
        hemopt.setdefault('reg_cov', 0.001)
        hemopt.setdefault('termmode', 'L')
        hemopt.setdefault('termvalue', 1e-5)
        hemopt.setdefault('max_iter', 100)
        hemopt.setdefault('min_iter', 1)
        hemopt.setdefault('sortclusters', 'f')
        hemopt.setdefault('initmode', 'baseem')
        #standard parameters (not usually changed)
        hemopt.setdefault('Nv', 100)
        hemopt.setdefault('tau', 10)
        hemopt.setdefault('initopt', {})
        hemopt['initopt'].setdefault('iter', 30)
        hemopt['initopt'].setdefault('trials', 4)
        hemopt.setdefault('inf_norm', 'nt')
        hemopt.setdefault('smooth', 1)
        #fixed parameters (not changed)
        hemopt['emit'] = {'type': 'gmm', 'covar_type': 'diag'}
        hemopt['M'] = 1

        emopt = {'trials': hemopt['trials']}

        # convert list of HMMs into an H3M
        H3M = hmms_to_h3m(hmms, 'diag')

        # run VHEM clustering
        h3m_out = hem_h3m_c(H3M, hemopt, emopt)

        # convert back to our format
        group_hmms = h3m_to_hmms(h3m_out,'diag')

        # sort clusters (false)
        #if hemopt['sortclusters']:
            #group_hmms = vbhmm_standardize(group_hmms, hemopt['sortclusters'])

        # save parameters
        group_hmms['hemopt'] = hemopt

        return group_hmms

#hmms_to_h3m(hmms,mode) (anche l'inversa, h3m to hmms)
def hmms_to_h3m(hmms,cov_mode):
        K=len(hmms) 
        omega = np.ones((1,K))/K
        nin = len(hmms[0]['pdf'][0]['mean'])
        h3m =[]
        h3m = {'K':K,'omega':omega,'hmm':[]}
        for j in range(K):
            tempHmm = {'emit':[]}
            s = len(hmms[j]['prior'])
            tempHmm['prior']=hmms[j]['prior']
            tempHmm['A']=hmms[j]['trans']
            for i in range(s):
                emit = {'type': 'gmm', 'nin': nin, 'ncentres': 1, 'priors': 1,'centres': hmms[j]['pdf'][i]['mean'].reshape(1, -1),
                                   'covar_type':'diag','covars':np.diag(hmms[j]['pdf'][i]['cov']).reshape(1,-1)}
                tempHmm['emit'].append(emit)
                print(emit['covars'])
            h3m['hmm'].append(tempHmm)
        return h3m
#h3m_to_hmms
def h3m_to_hmms(h3m,cov_mode):
        K = h3m['K']
        hmms = [] 
        for j in range(K):
            myhmm = h3m['hmm'][j] 
            tmphmm = {'prior':myhmm['prior'],'trans':myhmm['A'],'pdf':[] }
            for i in range(len(myhmm['emit'])):
                tempEmit = {'mean':myhmm['emit'][i]['centres'],'cov':np.diagflat(myhmm['emit'][i]['covars'])}
                tmphmm['pdf'].append(tempEmit)
            hmms.append(tmphmm)
        #assignment info
        group_hmms = {'Z':h3m['Z'],'LogLs':h3m['LogLs'],'LogL':h3m['LogL']}
        #get cluster assignments        
        maxZ = np.argmax(group_hmms['Z'], axis=1)
        group_hmms['label'] = maxZ
        #oppure group_hmms['label'] = maxZ.flatten()
        group_hmms['groups'] = {}
        group_hmms['group_size'] = []

        for j in range(len(hmms)):
            group_hmms['groups'][j] = np.where(group_hmms['label'] == j)[0]
            group_hmms['group_size'].append(len(group_hmms['groups'][j]))
            print(group_hmms['groups'][j])
        group_hmms['hmms']=hmms
        return group_hmms
#HEM_H3M_C
def hem_h3m_c(h3m_b, mopt, emopt):
        VERBOSE_MODE = mopt['verbose']
        LL_best = -np.inf
        re_do = 0

        if mopt['initmode'] in ['base', 'baseem']:
            while LL_best == -np.inf or np.isnan(LL_best):
                if VERBOSE_MODE == 1:
                    print('VHEM Trial: ')

                for t in range(emopt['trials']):
                    h3m_b1 = h3m_b.copy()

                    #if h3m_b['K'] > 400:
                        #inds = np.random.randint(h3m_b1['K'] - 400)
                        #new_idx = list(range(inds, inds + 400))
                        #h3m_b2 = {
                            #'omega': h3m_b['omega'][new_idx],
                            #'hmm': h3m_b['hmm'][new_idx],
                            #'K': 400
                        #}
                        #h3m_b2['omega'] /= np.sum(h3m_b2['omega'])
                        #h3m = initialize_hem_h3m_c(h3m_b2, mopt)
                    #else:
                    h3m = initialize_hem_h3m_c(h3m_b1, mopt)

                    h3m_new = hem_h3m_c_step(h3m, h3m_b, mopt)

                    if VERBOSE_MODE >= 2:
                        print('Trial', t, '\t - loglikelihood:', h3m_new['LogL'])
                    elif VERBOSE_MODE == 1:
                        print(t)

                    if t == 0:
                        LL_best = h3m_new['LogL']
                        h3m_out = h3m_new.copy()
                        t_best = t
                    elif h3m_new['LogL'] > LL_best:
                        LL_best = h3m_new['LogL']
                        h3m_out = h3m_new.copy()
                        t_best = t

                if (re_do <= 5) and (LL_best == -np.inf or np.isnan(LL_best)):
                    print('Need to do again... the LL was NaN ...')
                    re_do += 1
                elif (re_do <= 10) and (LL_best == -np.inf or np.isnan(LL_best)):
                    print("Use the 'gmm' instead of the", mopt['initmode'])
                    mopt['initmode'] = 'gmm'
                    re_do += 1
                elif (re_do > 10) and (LL_best == -np.inf or np.isnan(LL_best)):
                    print('\n\nGIVING UP ON THIS TAG!!!!!!\n\n')
                    h3m_out = h3m
                    h3m_out['given_up'] = 'too many trials'
                    LL_out = -np.finfo(float).eps
                    t_best = 0
                    break

        if VERBOSE_MODE >= 1:
            print('\nBest run is', t_best, ': LL =', LL_best, '\n')

        return h3m_out
#INITIALIZE HEM_H3M_C
def initialize_hem_h3m_c(h3m_b, mopt):
        #h3m_K = h3m_b['K']
        mopt['Nv'] = 1000 * h3m_b['K']

        T = mopt['tau']

        Kb = h3m_b['K']
        Kr = mopt['K']
        N = mopt['N']
        Nv = mopt['Nv']
        M = mopt['M']
        N=int(N)
        if mopt['initmode'] == 'baseem':
            h3m = {'K': Kr, 'hmm': []}
            for j in range(Kr):
                h3m['hmm'].append({'emit': [], 'prior': np.ones(N) / N, 'A': np.ones((N, N)) / N })
                for n in range(N):
                    randomb = np.random.randint(Kb)
                    randomg = np.random.randint(len(h3m_b['hmm'][randomb]['emit']))
                    h3m['hmm'][j]['emit'].append(h3m_b['hmm'][randomb]['emit'][randomg])
            h3m['omega'] = np.ones(Kr) / Kr
            h3m['LogL'] = -np.inf
            h3m['LogLs'] = [] 

        return h3m
#HEM_H3M_C_STEP 
def hem_h3m_c_step(h3m_r, h3m_b, mopt):
        num_iter = 0
    
        # Number of components in the base and reduced mixtures
        Kb = h3m_b['K']
        Kr = h3m_r['K']
    
        # Number of states
        N = h3m_b['hmm'][0]['A'].shape[0]
        Nr = h3m_r['hmm'][0]['A'].shape[0]
    
        # Number of mixture components in each state emission probability
        M = h3m_b['hmm'][0]['emit'][0]['ncentres']
    
 
        T = mopt['tau']

        dim = h3m_b['hmm'][0]['emit'][0]['nin']
    
        # Number of virtual samples
        virtualSamples = mopt['Nv']
    
        N_i = virtualSamples * h3m_b['omega'] * Kb
        N_i = N_i.reshape(-1, 1)
    
        if 'reg_cov' in mopt:
            reg_cov = mopt['reg_cov']
        else:
            reg_cov = 0
    
        if 'min_iter' not in mopt:
            mopt['min_iter'] = 0
    
    
        for j in range(Kr):
            for n in range(Nr):
                #h3m_r['hmm'][j]['emit'][n]['covars'] += reg_cov
                pass
           
        smooth = mopt['smooth']
    
        # Start looping variational E step and M step
        L_elbo = np.zeros((Kb,Kr))
        nu_1 = {}
        update_emit_pr = {}
        update_emit_mu = {}
        update_emit_M = {}
        sum_xi = {}
        while True:
        
            # E-step
        
            for i in range(Kb):
                hmm_b = h3m_b['hmm'][i]
                nu_1[i] = {}
                update_emit_pr[i] = {}
                update_emit_mu[i] = {}
                update_emit_M[i] = {}
                sum_xi[i] = {}
                for j in range(Kr):
                    hmm_r = h3m_r['hmm'][j]
                    result=hem_hmm_bwd_fwd(hmm_b, hmm_r, T, smooth)
                    L_elbo[i,j], nu_1[i][j], update_emit_pr[i][j], update_emit_mu[i][j], \
                    update_emit_M[i][j], sum_xi[i][j] = result
                    #print(result)
            #L_elbo /= inf_norm
            log_Z = (np.ones((Kb,1))*np.log(h3m_r['omega'])) + (N_i * np.ones((1, Kr))) * L_elbo
            Z = np.exp(log_Z - logtrick(log_Z.T).T[:,np.newaxis])
            new_LogLikelihood = np.sum(logtrick(log_Z.T).T)
        
            old_LogLikelihood = h3m_r['LogL']
            h3m_r['LogL'] = new_LogLikelihood
            h3m_r['LogLs'].append(new_LogLikelihood)
            h3m_r['Z'] = Z
            
            stop = False
        
            if num_iter > 1:
                changeLL = (new_LogLikelihood - old_LogLikelihood) / np.abs(old_LogLikelihood)
            else:
                changeLL = np.inf
        
            if changeLL < 0:
                print("The change in log likelihood is negative!!!")
        
            if 'termmode' in mopt and 'termvalue' in mopt:
                if mopt['termmode'] == 'L':
                    if changeLL < mopt['termvalue']:
                        stop = True
        
            if num_iter > mopt['max_iter']:
                stop = True

            
            if stop and num_iter >= mopt['min_iter']:
                break
        
            num_iter += 1
            old_LogLikelihood = new_LogLikelihood
            # M-step
        
            h3m_r_new = h3m_r.copy()

            # Re-estimation of the component weights (omega)
            omega_new = np.dot((np.ones((1, Kb)) / Kb), Z)
            print(omega_new.shape)
            h3m_r_new['omega'] = omega_new

            # Scale the Z_ij by the number of virtual samples
            Z = Z * (N_i * np.ones((1, Kr)))
            N2 = h3m_r['hmm'][j]['A'].shape[0]
           # print("zeta")
            #print(Z)
            #print("nu")

            for j in range(Kr):
                new_prior = np.zeros((N2, 1))
                new_A = np.zeros((N2, N2))
                new_Gweight = [np.zeros((1, M)) for _ in range(N2)]
                new_Gmu = [np.zeros((M, dim)) for _ in range(N2)]
                new_GMu = [np.zeros((M, dim)) for _ in range(N2)]

                for i in range(Kb):
                    if Z[i, j] > 0:
                        nu = nu_1[i][j]                 #1 by N vector
                        xi = sum_xi[i][j]              #N by N matrix (from - to) (andranno aggiunti dei : , : ,)
                        up_pr = update_emit_pr[i][j]     #N by M matrix
                        up_mu = update_emit_mu[i][j]     # N by dim by M matrix
                        up_Mu = update_emit_M[i][j]     #N by dim by M matrix

                        print(Z[i,j])
                        print(nu.shape)
                        print(Z[i,j]*nu)
                        new_prior = new_prior + Z[i, j] * nu[:,np.newaxis]
                        new_A += Z[i, j] * xi #

                        for n in range(N2):
                            new_Gweight[n] += Z[i, j] * up_pr[n, :] 

                            new_Gmu[n] = new_Gmu[n] + Z[i, j] * np.reshape(up_mu[n, :, :], (dim, -1)).T

                            if 'covar_type' in mopt['emit'] and mopt['emit']['covar_type'] == 'diag':
                                new_GMu[n] += Z[i, j] * np.reshape(up_Mu[n, :, :],(dim,-1)).T
                #print('new prior')
                #print(new_A)
                #print(new_prior.shape)
                #print(new_Gweight)
                #print(new_Gmu)
                #BUG FIX : if there are new prior or new statistic with zero components, all the other statistic will go to zero,
                #also in other trial
                if  np.any(new_prior == 0):
                    print("Non e andato come previsto!")
                    break
                #print(h3m_r_new['hmm'][j]['prior'])
                h3m_r_new['hmm'][j]['prior'] = new_prior / np.sum(new_prior)
                #print(h3m_r_new['hmm'][j]['prior'])
                #print(new_A)
                h3m_r_new['hmm'][j]['A'] = new_A / np.tile(np.sum(new_A, axis=1, keepdims=True), (1, N2))
                #print(h3m_r_new['hmm'][j]['A'])
                # ABC 2016-12-09 - save the counts in each emission
                h3m_r_new['hmm'][j]['counts_emit'] = np.sum(new_A, axis=0) + new_prior.T

                for n in range(N2):
                    h3m_r_new['hmm'][j]['emit'][n]['centres'] = new_Gmu[n]/ (new_Gweight[n].T * np.ones((1, dim)))
                    #print(h3m_r_new['hmm'][j]['emit'][n]['centres'])
                    if 'covar_type' in mopt['emit'] and mopt['emit']['covar_type'] == 'diag':

                        Sigma = new_GMu[n] - 2 * (new_Gmu[n] * h3m_r_new['hmm'][j]['emit'][n]['centres']) + \
                                (h3m_r_new['hmm'][j]['emit'][n]['centres']**2) * new_Gweight[n].T*np.ones((1, dim))
                      #  print(Sigma)
                        h3m_r_new['hmm'][j]['emit'][n]['covars'] = Sigma / new_Gweight[n].T*np.ones((1, dim))
                        h3m_r_new['hmm'][j]['emit'][n]['covars'] += reg_cov

                    h3m_r_new['hmm'][j]['emit'][n]['priors'] = new_Gweight[n] / np.sum(new_Gweight[n])
            
                   
            if  np.any(new_prior == 0):
                break
                    #EMISSION WITH 0 PRIO!
            ind_zero = np.where(h3m_r_new['omega'] == 0)[1]
           # print(ind_zero)
           # print(h3m_r_new['omega'].shape)
            for i_z in ind_zero:
                 print('!!! modifying h3m: one hmm has zero prior')
                 highest = np.argmax(h3m_r_new['omega'])
                 #print(highest)
                 h3m_r_new['omega'][0, i_z] = h3m_r_new['omega'][0, highest] / 2
                 h3m_r_new['omega'][0, highest] = h3m_r_new['omega'][0, highest] / 2
                 #normalize for safety
                 h3m_r_new['omega'] = h3m_r_new['omega'] / np.sum(h3m_r_new['omega'])
                 h3m_r_new['hmm'][i_z] = h3m_r_new['hmm'][highest].copy()
                 #perturb
                 h3m_r_new['hmm'][i_z]['prior'] = h3m_r_new['hmm'][highest]['prior'] + (.1/N) * np.random.rand(len(h3m_r_new['hmm'][highest]['prior']))
                 A = h3m_r_new['hmm'][highest]['A'].copy()
                 f_zeros = np.where(A == 0)
                 A = (.1/N) * np.random.rand(*A.shape)
                 A[f_zeros] = 0
                 h3m_r_new['hmm'][i_z]['A'] = A
                 #renormalize
                 h3m_r_new['hmm'][i_z]['prior'] = h3m_r_new['hmm'][i_z]['prior'] / np.sum(h3m_r_new['hmm'][i_z]['prior'])
                 h3m_r_new['hmm'][i_z]['A'] = h3m_r_new['hmm'][i_z]['A'] / np.tile(np.sum(h3m_r_new['hmm'][i_z]['A'], axis=1).reshape(-1, 1), (1, N))
            
            
            for j in range(Kr): 
                 for n in range(N2): 
                     ind_zero = np.where(h3m_r_new['hmm'][j]['emit'][n]['priors']==0)
                     for i_z in ind_zero:
                         ##print("modifying gmm emission : one component is zero")
                         highest = np.argmax(h3m_r_new['hmm'][j]['emit'][n]['priors'])
                         h3m_r_new['hmm'][j]['emit'][n]['priors'][i_z] = h3m_r_new['hmm'][j]['emit'][n]['priors'][highest] / 2
                         h3m_r_new['hmm'][j]['emit'][n]['priors'][highest] = h3m_r_new['hmm'][j]['emit'][n]['priors'][highest] / 2
                         #renormalize for safety
                         h3m_r_new['hmm'][j]['emit'][n]['priors'] = h3m_r_new['hmm'][j]['emit'][n]['priors'] / np.sum(h3m_r_new['hmm'][j]['emit'][n]['priors'])
                         h3m_r_new['hmm'][j]['emit'][n]['centres'][i_z, :] = h3m_r_new['hmm'][j]['emit'][n]['centres'][highest, :]
                         h3m_r_new['hmm'][j]['emit'][n]['covars'][i_z, :] = h3m_r_new['hmm'][j]['emit'][n]['covars'][highest, :]

                         centres_i_z = h3m_r_new['hmm'][j]['emit'][n]['centres'][i_z, :]
                         h3m_r_new['hmm'][j]['emit'][n]['centres'][i_z, :] = centres_i_z + 0.01 * np.random.rand(*centres_i_z.shape) * centres_i_z

           
        return h3m_r_new
###HEM HMM BWD FWD
def hem_hmm_bwd_fwd(hmm_b,hmm_r,T,smooth=1):   
    N=int(hmm_b['A'].shape[0])
    N2=hmm_r['A'].shape[0]
    M = hmm_b['emit'][0]['ncentres']
    dim = hmm_b['emit'][0]['nin']
    LLG_elbo,sum_w_pr,sum_w_mu,sum_w_Mu = g3m_stats(hmm_b['emit'],hmm_r['emit'])
    LLG_elbo /= smooth 
    ######################################
    ######## BACKWARD RECURSION ##########
    ######################################
    Ab = hmm_b['A']
    Ar = hmm_r['A']

    Theta = np.zeros((N2,N2,N,T))

    LL_old = np.zeros((N,N2))
    #print('Ar')
    #print(Ar)
    for t in range(T,1, -1):
        LL_new = np.zeros(LL_old.shape);
        for rho in range(N2):
            
            logtheta = np.log1p(Ar[rho, :])[:, np.newaxis] * np.ones((1, N)) + LLG_elbo.T + LL_old.T 
            logsumtheta = logtrick(logtheta)
            LL_new[:,rho]= Ab @ logsumtheta.T #@ corretto
            theta= np.exp( logtheta - logsumtheta); 
            Theta[rho,:,:,t-1] = theta
        LL_old = LL_new
    logtheta = np.log1p(hmm_r['prior']*np.ones((1, N))) + LLG_elbo.T + LL_old.T
    logsumtheta = logtrick(logtheta)
    LL_elbo = np.dot(hmm_b['prior'].T, logsumtheta.T) 

    theta = np.exp(logtheta - logsumtheta)

    Theta_1 = theta
    #####################################
    ######## FORWARD RECURSION ##########
    #####################################

    #nu11 = np.dot(np.ones((N2,1)),hmm_b['prior'].T) * Theta_1 #####
    nu = np.ones((N2,1))@hmm_b['prior'].T * Theta_1
    sum_nu_1 = np.sum(nu,axis=1).T
    sum_t_nu = nu
    foo = np.dot(nu , Ab)

    sum_t_sum_g_xi = np.zeros((N2,N2))

    for t in range(2,T):

        for sigma in range(N2):

            xi_foo = foo * np.reshape(Theta[:,sigma,:,t], (Theta.shape[0], Theta.shape[2])) #epsilon
            sum_t_sum_g_xi[:,sigma] = sum_t_sum_g_xi[:,sigma] + np.sum(xi_foo,axis=1)  #
            nu[sigma, :] = np.sum(xi_foo, axis=1)
        sum_t_nu = sum_t_nu + nu
    sum_xi = sum_t_sum_g_xi
    #CALCULATE STATISTICS
    update_emit_pr = np.zeros((N2,M))
    update_emit_mu = np.zeros((N2,dim,M))
    update_emit_Mu = np.zeros((N2,dim,M))
    for sigma in range(N2):

        update_emit_pr[sigma,:]= sum_t_nu[sigma,:] @ np.array(sum_w_pr[sigma]) #
        foo_sum_w_mu = sum_w_mu[sigma] 
        foo_sum_w_Mu = sum_w_Mu[sigma] 

        for l in range(M):
            update_emit_mu[sigma,:,l] = sum_t_nu[sigma,:] @ foo_sum_w_mu[:,:,l] # 
            update_emit_Mu[sigma,:,l] = sum_t_nu[sigma,:] @ foo_sum_w_Mu[:,:,l] # 

    return LL_elbo, sum_nu_1, update_emit_pr,update_emit_mu,update_emit_Mu,sum_xi

#G3M_STATS
def g3m_stats(g3m_b, g3m_r):
        
        N = len(g3m_b)
        N2 = len(g3m_r)
        M = g3m_b[0]['ncentres']
        dim = g3m_b[0]['nin']

        LLG_elbo = np.zeros((N, N2))
        sum_w_pr = []
        sum_w_mu = []
        sum_w_Mu = []

        for rho in range(N2):
            foo_sum_w_pr = np.zeros((N, M))
            foo_sum_w_mu = np.zeros((N, dim, M))
            foo_sum_w_Mu = np.zeros((N, dim, M))

            gmmR = g3m_r[rho]

            for beta in range(N):
                gmmB = g3m_b[beta]
                #compute the expected log-likelihood between the Gaussian components
                #E_M(b),beta,m [log p(y | M(r),rho,l)], for m and l 1 ...M
                ELLs = compute_exp_lls(gmmB, gmmR)

                #compute log(omega_r) + E_M(b),beta,m [log p(y | M(r),rho,l)]
                log_theta = ELLs + np.log(gmmR['priors'])
                ##adv maybe da correggere compute log Sum_b omega_b exp(-D(fa,gb))
                log_sum_theta = logtrick(log_theta.T).T#LOGTRICK USED
                # compute L_variational(M(b)_i,M(r)_j) = Sum_a pi_a [  log (Sum_b omega_b exp(-D(fa,gb)))]
                LLG_elbo[beta, rho] = gmmB['priors']* log_sum_theta
                
                theta = np.exp(log_theta) #capire cosa fare con gliones
                foo_sum_w_pr[beta, :] = gmmB['priors']* theta

                foo_sum_w_mu[beta, :, :] = (( gmmB['priors'] * theta.T)*gmmB['centres']).T
                
                foo_sum_w_Mu[beta, :, :] = ((gmmB['priors'] * theta.T) *(gmmB['centres']**2 + gmmB['covars']) ).T
            sum_w_pr.append(foo_sum_w_pr)
            sum_w_mu.append(foo_sum_w_mu)
            sum_w_Mu.append(foo_sum_w_Mu)
        #print(sum_w_pr)
        #print(sum_w_mu)
        return LLG_elbo, sum_w_pr, sum_w_mu, sum_w_Mu
#compute exp gll
def compute_exp_lls(gmmA, gmmB):
        dim = gmmA['nin']
        A = gmmA['ncentres']
        B = gmmB['ncentres']
        ELLs = np.zeros((A, B))
        for a in range(A):
            for b in range(B):
                covar_type = gmmA['covar_type']
                #print(gmmA['covars'][a, :])
                if covar_type == 'diag':
                    ELLs[a, b] = -0.5 * (dim * np.log(2 * np.pi) + np.sum(np.log(gmmB['covars'][b, :])) + np.sum(gmmA['covars'][a, :] / gmmB['covars'][b, :]) + np.sum((( gmmA['centres'][a, :]-gmmB['centres'][b, :]) ** 2) / gmmB['covars'][b, :]))
                else:
                    raise ValueError('Covariance type not supported')
        
        return ELLs


#######SETDEFAULT#######(COMPLETA)
def setdefault(vbopt, field, value):
        if field not in vbopt:
            vbopt[field] = value
        return vbopt

def logtrick(lA):
    # Calcola il massimo lungo ciascuna colonna
    mv = np.max(lA, axis=0)
    
    # Sottrae il massimo da ogni elemento nella matrice
    temp = lA - mv
    
    # Calcola la somma esponenziale lungo l'asse delle righe
    cterm = np.sum(np.exp(temp), axis=0)
    
    # Calcola il risultato finale
    s = mv + np.log(cterm)
    
    return s
