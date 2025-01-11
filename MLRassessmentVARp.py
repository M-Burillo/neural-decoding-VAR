#Copyright 2024 Marc Burillo

#This code is not authorized for use, copying, modification, or distribution without explicit permission from the author.

#This code implements the multi logistic regression of sklearn to assess if the VAR(p) characterization matches the 5 labeled states. This code requires to run before multitrialVARp.py. The output is the accuracy in trainset (called precision) and in testset. Also the Confussion Matrices for each subset. The CV method is 10-Fold

import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

options = np.loadtxt('channels.txt').astype(int) #options are the channels included in each VAR(p) model: from 1 to 5 channels explored in a same model in our project. See channels.txt for an example
if len(np.shape(options)) == 1 
	options.reshape(1,-1)
orders = np.loadtxt('orders.txt').astype(int) #p0 are the optimal ordered determined with: order_determination.py for each model in channels. See orders.txt for an example

folding=np.load('10_10fold.npy') #10 random 10-folds over 310 trials (time-series recorded with equivalent conditions): 10 sets of 31 elements randomly chosen among 0 and 309.
nrep=np.shape(folding)[0]
nfoldings=np.shape(folding)[1]
txfolding=np.shape(folding)[2]
nstates=5 #determined by our experiment

#--------------------------
def get_kth_fold(data, N, k):
	# Calculate the size of each fold
	fold_size = len(data) // N

	# Ensure that k is within the valid range
	if k < 0 or k >= N:
	raise ValueError("The value of k must be between 0 and N-1")

	# Calculate the start and end indices of the k-th fold
	start = k * fold_size
	end = start + fold_size

	# Return the k-th fold
	kth_fold = data[start:end]

	return kth_fold
#-------------------------------

def exclude_kth_fold(data, N, k):
	# Calculate the size of each fold
	fold_size = len(data) // N

	# Ensure that k is within the valid range
	if k < 0 or k >= N:
	raise ValueError("The value of k must be between 0 and N-1")

	# Calculate the start and end indices of the k-th fold
	start = k * fold_size
	end = start + fold_size

	# Return the elements outside of the k-th fold
	remaining_data = np.concatenate((data[:start], data[end:]))

	return remaining_data
    
    
for counter, channels in enumerate(options):
	channels = [channels]
	print(channels)
	order = orders[counter]
	p0=order
	Xname = 'XVARdataset_order'+str(p0)+'_channels'
	Xname += "_" + "_".join(np.array(channels).astype(str))
	Xname += '.npy'
	Yname = 'YVARdataset_order'+str(p0)+'_channels'
	Yname += "_" + "_".join(np.array(channels).astype(str))
	Yname += '.npy'
	M_ar = np.load(Xname)
	label_ar=np.load(Yname)

	p0=order
#	for order in range(p0, p1):
	for rep in range(nrep):
		for fold in range(nfoldings):
			X_train = exclude_kth_fold(M_ar[rep * nfoldings * txfolding * nstates:(rep + 1) * nfoldings * txfolding * nstates], nfoldings, fold)
			Y_train = exclude_kth_fold(label_ar[rep * nfoldings * txfolding * nstates:(rep + 1) * nfoldings * txfolding * nstates], nfoldings, fold)
			X_test = get_kth_fold(M_ar[rep * nfoldings * txfolding * nstates:(rep + 1) * nfoldings * txfolding * nstates], nfoldings, fold)
			Y_test = get_kth_fold(label_ar[rep * nfoldings * txfolding * nstates:(rep + 1) * nfoldings * txfolding * nstates], nfoldings, fold)

			model = LogisticRegression(penalty=tipus, C=c, solver='saga', max_iter=5000)
			model.fit(X_train, Y_train)

			Y_pred = model.predict(X_test)
			accuracy[order - p0][rep * nfoldings + fold] = accuracy_score(Y_test, Y_pred)

			if rep == 0:
				cm = [confusion_matrix(Y_test, Y_pred, labels=model.classes_)]
			else:
				cm = np.vstack((cm, [confusion_matrix(Y_test, Y_pred, labels=model.classes_)]))

			Y_pred = model.predict(X_train)
			precision[order - p0][rep * nfoldings + fold] = accuracy_score(Y_train, Y_pred)

			if rep == 0:
				cm2 = [confusion_matrix(Y_train, Y_pred, labels=model.classes_)]
			else:
				cm2 = np.vstack((cm2, [confusion_matrix(Y_train, Y_pred, labels=model.classes_)]))

	cm = np.mean(cm, axis=0)
	row_sums = cm.sum(axis=1)
	cm = cm / row_sums[:, np.newaxis]

	disp = ConfusionMatrixDisplay(cm)
	disp.plot(values_format='.2f')
	disp.ax_.set_title('AR(' + str(order) + ') Testset| Acc. ' + "{:.2f}$\pm${:.2f}".format(np.mean(accuracy[order - p0]), np.std(accuracy[order - p0])))
	disp.figure_.savefig('Results/ARconfusion_matrix_5_states_50L10_order' +str(order)+'ch'+str(options[counter])+ 'Testvf2SA.jpeg', bbox_inches='tight', dpi=300)

	cm2 = np.mean(cm2, axis=0)
	row_sums = cm2.sum(axis=1)
	cm2 = cm2 / row_sums[:, np.newaxis]

	disp = ConfusionMatrixDisplay(cm2)
	disp.plot(values_format='.2f')
	disp.ax_.set_title('AR(' + str(order) + ') Trainset| Prec. ' + "{:.2f}$\pm${:.2f}".format(np.mean(precision[order - p0]), np.std(precision[order - p0])))
	disp.figure_.savefig('Results/ARconfusion_matrix_5_states_50L10_order'  +str(order)+'ch'+str(options[counter])+'Trainvf2SA.jpeg', bbox_inches='tight', dpi=300)
	code = 'order'+str(p0)+'_channels'
	code += "_" + "_".join(np.array(channels).astype(str)
	np.save('VARcmT'+code+'.npy', cm2)
	np.save('ARcmTT'+code+'.npy', cm)
	np.save('VARaccT'+code+'.npy', accuracy)
	np.save('VARprecT'+code+'.npy', precision)
