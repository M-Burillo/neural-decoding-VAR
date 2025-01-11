#Copyright 2024 Marc Burillo

#This code is not authorized for use, copying, modification, or distribution without explicit permission from the author.

#This code determines the optimal order (p) to use in a VAR(p) model. It requires to have determined the channels considered in each model and the CV method used and the processed time-series

import numpy as np
import sys
import matplotlib.pyplot as plt

options = np.loadtxt('channels.txt').astype(int) #options are the channels included in each VAR(p) model: from 1 to 5 channels explored in a same model in our project. See channels.txt for an example
if len(np.shape(options)) == 1 
    options.reshape(1,-1)

rel = np.load('310_filtered.npy') #dataset of processed time series: structure (310, 5, 508, 256) = (trials, states, timepoint, channel)


folding=np.load('10_10fold.npy') #10 random 10-folds over 310 trials (time-series recorded with equivalent conditions): 10 sets of 31 elements randomly chosen among 0 and 309.

#this parameters can be changed, but this is already computationally expensive
nrep=10
nfoldings=1
txfolding=1


nstates=5 #determined by our experiment




# VAR(p) model fitting functions according to the Dissertation included in the GitHub but see also: 
#Helmut Lutkepohl. New Introduction to Multiple Time Series Analysis. en. Berlin, Heidelberg: Springer, 2005. isbn: 978-3-540-40172-8 978-3-540-27752-1. Chapter 3.4, following same notation


def capZt2(vslice,p,t,y, trial, state):
    if t==0:
        Zt = np.insert(y[trial][state][t+p-1::-1,vslice],0,1)
    else:
        Zt = np.insert(y[trial][state][t+p-1:t-1:-1,vslice],0,1)
    return Zt.flatten('F')

#----------------------------
def capZ2(vslice,p,T,y, trial, state):
    Z = capZt2(vslice,p, 0,y, trial[0],state)

    for j in trial:
        if j == trial[0]:
            for i in range(1,T-p):
                Z = np.vstack((Z,capZt2(vslice,p, i,y, j,state)))
        else:
            for i in range(0,T-p):
                Z = np.vstack((Z,capZt2(vslice,p, i,y, j,state)))
    return Z
#----------------------------
def capYn2(vslice,p,T,y, trial, state):
    Y = y[trial][state][p:T,vslice]
    return Y.transpose()
#---------------------------
def capY2(vslice,p,T,y, vtrials, state):
    Y = capYn2(vslice,p, T,y, vtrials[0],state)
    for j in range(1,np.size(vtrials)):
        Y = np.hstack((Y,capYn2(vslice,p, T,y, vtrials[j],state)))
    return Y.transpose()
#--------------------------
def hatbetaN(vslice,p,T,y, vtrials,state):
    return np.linalg.lstsq(capZ2(vslice,p,T,y,vtrials,state), capY2(vslice,p,T,y, vtrials, state),rcond=1e-15)[0].transpose()   
#---------------------------
def VAR_coefN(vslice,p,T,y,vtrials,state):
    beta2 = hatbetaN(vslice,p,T, y,vtrials,state)
    beta = beta2.flatten('F')
    k = np.size(vslice)
    mu = beta[0:k].flatten()
    beta = np.reshape(beta, (np.size(beta),1))
    for j in range(p):
        A = np.reshape(beta[k+k*k*j:k+k*k*(j+1)].reshape(1,-1),(k,k))
        if j == 0:
            Ap = [A]
        else:
            Ap = np.concatenate((Ap,[A]), axis = 0)
    return beta2, mu,np.array(Ap)
#-----------------------------
def hatbetaN2(Z,Y):
    return np.linalg.lstsq(Z, Y,rcond=1e-15)[0].transpose()
#-----------------------------
def VAR_coefN2(vslice,p,T,y,vtrials,state,Z,Y):
    beta2 = hatbetaN2(Z,Y)
    beta = beta2.flatten('F')
    k = np.size(vslice)
    mu = beta[0:k].flatten()
    beta = np.reshape(beta, (np.size(beta),1))
    for j in range(p):
        A = np.reshape(beta[k+k*k*j:k+k*k*(j+1)].reshape(1,-1),(k,k))
        if j == 0:
            Ap = [A]
        else:
            Ap = np.concatenate((Ap,[A]), axis = 0)
    return beta2, mu,np.array(Ap)
#----------------------------
def tildeS(T2,Y,B,Z):
    aux = Y-np.matmul(B,Z)
    return (1/T2)*np.matmul(aux, aux.transpose())
#-----------------------------
def logL(vslice, p, T,y, vtrials, state, Y,B, Z, hS):
    N = np.size(vtrials)
    K = np.size(vslice)
    Sinv = np.linalg.inv(hS)
    T -=p
    return -K*N*T/2*np.log(2*np.pi)+N*T/2*np.log(np.linalg.det(Sinv))-(1/2)*T*N*K
#---------------------------
def BIC(k,n,L):
    return k*np.log(n)-2*L
#-----------------------------
def get_specific_fold(n, exclude):
    return np.arange(n)[np.r_[0:exclude, exclude+1:n]]


#-------------------------------------


#----------------Multi trial fitting through VAR(p)--------------------

for counter, channels in enumerate(options):
    for order in range(1,40):
    	string = 'VAR_order' + str(p0) + '_channels'
    	string += "_" + "_".join(np.array(channels).astype(str))
    	string += '/'
	p0 = order
        for rep in range(nrep):
            for fold in range(nfoldings):
                print(rep)
                print(fold)
                for mtrial in range(txfolding):
                    for state in range(nstates):
                        vtrials = np.array(get_specific_fold(folding[rep, fold, :], mtrial)).astype(int)  # Which trials do you want to use in your regression?
                        T = np.shape(rel)[2]  # How many time-points are in a single channel in a given trial and state
                        Y = capY2(channels, order, T, rel, vtrials, state).transpose()
                        Z = capZ2(channels, order, T, rel, vtrials, state).transpose()
                        B, nu, Ap = VAR_coefN2(channels, order, T, rel, vtrials, state, Z.transpose(), Y.transpose())  # mu is the mean, Ap is the set of matrix of VAR(k,p)
                        #hS = tildeS(np.size(vtrials) * (T - order), Y, B, Z) we dont use the estimated covariance matrix of the noise, although it could be added
                        np.save(string + 'B_order' + str(order) + 'rep' + str(rep) + 'fold' + str(fold) + 'mtrial' + str(mtrial) + 'state' + str(state) +'.npy', B)      
                        L = logL(channels, order, T, rel, vtrials, state, Y, B, Z, hS)
                        np.save(string + 'L_order' + str(order) + 'rep' + str(rep) + 'fold' + str(fold) + 'mtrial' + str(mtrial) + 'state' + str(state) +'.npy', L)


def BIC(k,n,L):
    return k*np.log(n)-2*L


bic = np.zeros((40-1,nrep*nfoldings*txfolding,len(options),5))
logL= np.zeros((40-1,nrep*nfoldings*txfoldin,len(options),5))

for counterch, channels in enumerate(options):
    for counter, order in enumerate(range(1, 40)):
        # Construct the base string
        string = 'VAR_order' + str(order) + '_channels'
        string += "_" + "_".join(np.array(channels).astype(str))
        string += '/'

        for rep in range(nrep):
            for fold in range(nfoldings):
                for mtrial in range(txfolding):
                    for state in range(nstates):
                        # Load the data
                        L = np.load(string + 'L_order' + str(order) + 
                                    'rep' + str(rep) + 
                                    'fold' + str(fold) + 
                                    'mtrial' + str(mtrial) + 
                                    'state' + str(state) + 
                                    'channel' + "_".join(map(str, channels)) + '.npy')
                        
                        T = 518 - order
                        K = len(channels)
                        N = 30  # Fixed by the multi-trial fitting
                        
                        # Update the BIC and log-likelihood arrays
                        bic[counter, fold, counterch, state] = BIC(K * K * order + K, N * K * T, L)
                        logL[counter, fold, counterch, state] = L

# Save results
np.save('all_logL.npy', logL)
np.save('all_BIC.npy', bic)

# Elbow calculation function
def elbow(BIC_values, min_change):
    total = np.abs(BIC_values[0] - BIC_values[-1])
    for i in range(len(BIC_values)):
        print('new')
        print(BIC_values[i])
        print(BIC_values[-1] + min_change * total)
        if BIC_values[i] < BIC_values[-1] + min_change * total:
            return 2 * i + 2

# Initialize arrays to store optimal orders
orderbic = np.zeros((len(options), 5))
orderlogL = np.zeros((len(options), 5))

# Iterate over options and states to calculate optimal orders
for counterch, channels in enumerate(options):
    for state in range(5):  # Assuming states are indexed from 0 to 4
        print(f"Processing channels: {channels}, state: {state}")

        # Calculate optimal orders using BIC
        mean_bic = np.mean(bic[:, :, counterch, state], axis=1)
        orderbic[counterch, state] = elbow(mean_bic, 0.15)

        # Calculate optimal orders using log-likelihood differences
        mean_logL_diff = (np.mean(logL[1:, :, counterch, state], axis=1) - 
                          np.mean(logL[:-1, :, counterch, state], axis=1))
        orderlogL[counterch, state] = elbow(mean_logL_diff, 0.15)
np.save('all_orderlogL.npy', orderlogL)
np.save('all_orderbic.npy', orderbic)

with open("orders.txt", "w") as file:
    for value in np.max(orderbic,axis=1):
        file.write(f"{value}\n")  # Write each value on a new line

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=10)  #  titol de verita


colors=['blue', 'orange', 'green', 'red', 'black']


for k in options:
    figure, axis =plt.subplots(1,1, figsize=(9,6))
    yaxis = axis.twinx()
    for j in range(5):
            axis.errorbar(orders,np.mean(bic[:,:,k, j],axis=1),yerr=np.std(bic[:,:,k,j],axis=1), ms=3, fmt = 'o',capsize=3, c=colors[j],alpha=0.5, label ='s='+str(j+1))
            #axis[int(k/2),k%2].plot([orders[np.argmax(-np.mean(bic[:,:,k, j],axis=1))],orders[np.argmax(-np.mean(bic[:,:,k, j],axis=1))]],[np.min(np.mean(bic[:,:,k, j],axis=1)),np.max(np.mean(bic[:,:,k, j],axis=1))],c=colors[j], ls='dashed')
            #yaxis.errorbar(orders[1:],np.mean(logL[1:,:,k, j],axis=1)-np.mean(logL[0:np.size(orders)-1,:,k, j],axis=1),yerr=np.std(logL[1:,:,k, j],axis=1)+np.std(logL[0:np.size(orders)-1,:,k, j],axis=1), ms=3, fmt = 'o',capsize=3, c=colors[j],alpha=0.5)
            yaxis.plot([2,50],[5.991,5.991],c='black', ls='dotted')
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    yaxis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axis.set_xlabel('Order $p$')
    yaxis.set_ylabel(r'$\Lambda$')
    axis.set_ylabel('BIC')
    axis.set_xticks(np.arange(0,50,2))
    axis.grid(visible=True, which='both', axis='both')

    figure.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figure.tight_layout(pad=0.001)
    code = '_channels'
    code += "_" + "_".join(np.array(channels).astype(str))
 
    figure.savefig("ARorderSelectionBIC"+code +".pdf",bbox_inches='tight')

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=10)  #  titol de verita

for k in options:
    figure, axis =plt.subplots(1,1, figsize=(9,6))
    yaxis = axis.twinx()
    for j in range(5):
            #axis.errorbar(orders,np.mean(bic[:,:,k, j],axis=1),yerr=np.std(bic[:,:,k,j],axis=1), ms=3, fmt = 'o',capsize=3, c=colors[j],alpha=0.5, label ='s='+str(j+1))
            #axis[int(k/2),k%2].plot([orders[np.argmax(-np.mean(bic[:,:,k, j],axis=1))],orders[np.argmax(-np.mean(bic[:,:,k, j],axis=1))]],[np.min(np.mean(bic[:,:,k, j],axis=1)),np.max(np.mean(bic[:,:,k, j],axis=1))],c=colors[j], ls='dashed')
            axis.errorbar(orders[1:],np.mean(logL[1:,:,k, j],axis=1)-np.mean(logL[0:np.size(orders)-1,:,k, j],axis=1), yerr=np.std(logL[1:,:,k, j],axis=1)+np.std(logL[0:np.size(orders)-1,:,k, j],axis=1), ms=3, fmt = 'o',capsize=3, c=colors[j],alpha=0.5, label ='s='+str(j+1))
            axis.plot([2,50],[5.991,5.991],c='black', ls='dotted')
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    yaxis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axis.set_xlabel('Order $p$')
    yaxis.set_ylabel(r'$\Lambda$')
    axis.set_ylabel('BIC')
    axis.set_xticks(np.arange(0,50,2))
    axis.grid(visible=True, which='both', axis='both')

    figure.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figure.tight_layout(pad=0.001)
    code = '_channels'
    code += "_" + "_".join(np.array(channels).astype(str))
 
    figure.savefig("ARorderSelectionLOGL"+code +".pdf",bbox_inches='tight')
