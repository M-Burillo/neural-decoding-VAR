#Copyright 2024 Marc Burillo

#This code is not authorized for use, copying, modification, or distribution without explicit permission from the author.

#This code implements the multi-trial fitting of a VAR(p) model. It requires to have determined the channels considered in each model and the order of the model, the CV method used and the processed time-series

import numpy as np
import sys
import os

options = np.loadtxt('channels.txt').astype(int) #options are the channels included in each VAR(p) model: from 1 to 5 channels explored in a same model in our project. See channels.txt for an example
if len(np.shape(options)) == 1:
    options.reshape(1,-1)
orders = np.loadtxt('orders.txt').astype(int) #p0 are the optimal ordered determined with: order_selection.py for each model in channels.

rel = np.load('310_filtered.npy') #dataset of processed time series: structure (310, 5, 508, 256) = (trials, states, timepoint, channel)


folding=np.load('10_10fold.npy') #10 random 10-folds over 310 trials (time-series recorded with equivalent conditions): 10 sets of 31 elements randomly chosen among 0 and 309.
nrep=np.shape(folding)[0]
nfoldings=np.shape(folding)[1]
txfolding=np.shape(folding)[2]
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
    p0 = orders[counter]
    for order in range(p0,p0+1): #for only 1 order is unnecessary, but itmay be desirable to explore different orders
    	string = 'VAR_order' + str(p0) + '_channels'
    	string += "_" + "_".join(np.array(channels).astype(str))
    	if not os.path.exists(string):
	    os.mkdir(string)
    	string += '/'

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


def stringtrain(order,rep,fold,mtrial,state,channels):
    nstring = string
    nstring += 'B_order' + str(order) + 'rep' + str(rep) + 'fold' + str(fold) + 'mtrial' + str(mtrial) + 'state' + str(state)+ 'channels'
    nstring += "_" + "_".join(np.array(channels).astype(str))
    nstring += '.npy'
    return nstring
    
#WRAPPING ALL THE MODELS IN A SINGLE ONE FILE, BASED ON THE BRAIN STATE/EXPERIMENT STAGE
for counter, channels in enumerate(options):
    for order in range(p0, p1):
    	string = 'VAR_order' + str(p0) + '_channels'
    	string += "_" + "_".join(np.array(channels).astype(str))
	
    	string += '/'
	order = orders[counter]
	p0 = order
	for rep in range(nrep):
		for fold in range(nfoldings):
			for mtrial in range(txfolding):
				for state in range(nstates):
					nameM = stringtrain(order, rep, fold, mtrial, state, channels)
					if rep ==0 and fold == 0 and mtrial == 0 and state == 0:
					    M_VAR = np.load(nameM).flatten('F')
					    label_VAR = [state]
					else:
					    M_VAR = np.vstack((M_ar,np.hstack((np.load(nameM).flatten('F'),np.load(nameSA).flatten('F')))))
					    label_VAR.append(state)
	label_VAR = np.array(label_ar)
	Xname = 'XVARdataset_order'+str(p0)+'_channels'
	Xname += "_" + "_".join(np.array(channels).astype(str))
	Xname += '.npy'
	Yname = 'YVARdataset_order'+str(p0)+'_channels'
	Yname += "_" + "_".join(np.array(channels).astype(str))
	Yname += '.npy'
	np.save(Xname, M_VAR)
	np.save(Yname, label_VAR)
