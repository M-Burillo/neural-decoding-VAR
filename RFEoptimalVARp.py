#Copyright 2024 Marc Burillo

#This code is not authorized for use, copying, modification, or distribution without explicit permission from the author.

#This code implements the Recursive Feature Elimination (RFE) algorithm on the VAR(p) characterization matches. This code requires to run before MLRassessmentVARp.py. The output is the optimal features over a training process with the CV method is 10-Fold, when using RFE for a smaller subset of features. It also reports the accuracy and precision distributions. The RFE method is implemented recursively: in each individual training process, only the previous surviving features are re-analysed



import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


options = np.loadtxt('channels.txt').astype(int) #options are the channels included in each VAR(p) model: from 1 to 5 channels explored in a same model in our project. See channels.txt for an example
if len(np.shape(options)) == 1 
	options.reshape(1,-1)
orders = np.loadtxt('orders.txt').astype(int) #p0 are the optimal ordered determined with: order_determination.py for each model in channels. See orders.txt for an example

folding=np.load('10_10fold.npy') #10 random 10-folds over 310 trials (time-series recorded with equivalent conditions): 10 sets of 31 elements randomly chosen among 0 and 309.
nrep=np.shape(folding)[0]
nfoldings=np.shape(folding)[1]
txfolding=np.shape(folding)[2]
nstates=5 #determined by our experiment

for channels in options:
	for order in range(p0, p1):
		Xname = 'XVARdataset_order'+str(p0)+'_channels'
		Xname += "_" + "_".join(np.array(channels).astype(str))
		Xname += '.npy'
		Yname = 'YVARdataset_order'+str(p0)+'_channels'
		Yname += "_" + "_".join(np.array(channels).astype(str))
		Yname += '.npy'
		M_ar = np.load(Xname)
		label_ar=np.load(Yname)
		
		n_features = len(channels)*len(channels)*order+len(channels)
		if p0>n_features > 50: 
			size_rfe=np.concatenate((np.arange(1,10),np.arange(10,n_features,10))).astype(int) #multi-channel model
		else:
			size_rfe=np.concatenate((np.arange(1º,10),np.arange(10,n_features,2))).astype(int) #single-channel model

		glob_acc = np.zeros((len(size_rfe), nrep*nfoldings))
		glob_prec =np.zeros((len(size_rfe), nrep*nfoldings))
		global_optimal_features = np.zeros((len(size_rfe), nrep*nfoldings, n_features))
		
		for rep in range(nrep):
			for fold in range(nfoldings):
				prev_features = np.arange(n_features).astype(int) #we start with all features
				for index, aimed_features in enumerate(size_rfe[::-1]):
					X_train = exclude_kth_fold(M_ar[rep * nfoldings * txfolding * nstates:(rep + 1) * nfoldings * txfolding * nstates], nfoldings, fold)
					Y_train = exclude_kth_fold(label_ar[rep * nfoldings * txfolding * nstates:(rep + 1) * nfoldings * txfolding * nstates], nfoldings, fold)
					X_train[prev_features>=0] #select from the previous existing features
					X_test = get_kth_fold(M_ar[rep * nfoldings * txfolding * nstates:(rep + 1) * nfoldings * txfolding * nstates], nfoldings, fold)
					X_test[prev_features>=0]
					Y_test = get_kth_fold(label_ar[rep * nfoldings * txfolding * nstates:(rep + 1) * nfoldings * txfolding * nstates], nfoldings, fold)

					rfe = RFE(estimator=LogisticRegression(penalty=tipus, C=c, solver='saga', max_iter=5000), n_features_to_select=aimed_features)
					model = LogisticRegression(penalty=tipus, C=c, solver='saga', max_iter=5000)
					pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
					pipeline.fit(X_train, Y_train)

					Y_pred = pipeline.predict(X_test)
					global_acc[index,rep * nfoldings + fold] = accuracy_score(Y_test, Y_pred)
					Y_pred = pipeline.predict(X_train)
					global_prec[index,wrep * nfoldings + fold] = accuracy_score(Y_train, Y_pred)
					
					for i in range(X_test.shape[1]):
						if rfe.support_[i]:
							global_optimal_features[index, rep * nfoldings + fold,i] = 1 #this feature is optimal given this size of aimed_features
						else:
							prev_features[i] = -1 #this feature was eliminated


		code = 'order'+str(p0)+'_channels'
		code += "_" + "_".join(np.array(channels).astype(str)
		np.save('RFEacc'+code+'.npy', global_acc)
		np.save('RFEprec'+code+'.npy', global_prec)
		np.save('RFEoptimal_features'+code+'.npy', global_optimal_features)
