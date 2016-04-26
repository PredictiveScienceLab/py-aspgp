#!/usr/bin/env python

"""
Demonstration with synthetic example.

Number of active dimensions - 1.

Number of input dimensions - 20.

Number of training samples - 80.

Number of test samples - 20.

Gaussian noise variance added to the output - 0.01


AUTHOR - Rohit Tripathy
"""

#import relevant modules
import numpy as np
import GPy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
from pdb import set_trace as keyboard
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import os
import sys
cwd=os.getcwd()
aspgpdir=os.path.abspath(os.path.join(cwd, '..'))
syndatadir=os.path.abspath(os.path.join(cwd, 'synthetic_data'))
sys.path.insert(0, aspgpdir)
import aspgp
import pickle


#Do anything at all only if the model file doesnt aleady exist
modelfilepath=os.path.join(syndatadir, 'syn_model_1d.pcl')
if not os.path.exists(modelfilepath):
    #load the full dataset 
    Xfile=os.path.join(syndatadir, 'X.npy')
    Yfile=os.path.join(syndatadir, 'Y.npy')
    X=np.load(Xfile)
    Y=np.load(Yfile)
    
    #shuffle dataset and split the dataset into train test sets
    X, Y = shuffle(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=80)
    
    #save the training, test data
    np.save(os.path.join(syndatadir, 'X_train.npy'), X_train)
    np.save(os.path.join(syndatadir, 'Y_train.npy'), Y_train)
    np.save(os.path.join(syndatadir, 'X_test.npy'), X_test)
    np.save(os.path.join(syndatadir, 'Y_test.npy'), Y_test)
    
    #NOTE: If you want to generate models for a range of
    #       active dimensions rather than a single active dimension
    #       uncomment the list definition below and loop over it.
    
    #active_dim=[1, 2]
    
    #define an inner kernel
    active_dim=1
    inner_kernel=GPy.kern.Matern32(active_dim)
    
    #define the aspgp model
    model=aspgp.ActiveSubspaceGPRegression(X_train, Y_train, inner_kernel)
    
    #define a dictionary defining the parameters for Stiefel optimization
    stiefel_opt = {'disp': False,
                   'tau_max': 0.5,
                   'tol': 1e-3,
                   'tau_find_freq': 10,
                   'max_it': 1}
    
    #Display the model as it is currently
    print "="*20
    print "Before optimization:"
    print "="*20
    print model
    print "="*100
    
    
    #define the parameters of the overall optimization process.
    num_restarts = 10
    tol=1e-3
    disp='True'
    max_it=200
    
    #optimize the model.
    model.optimize_restarts(max_it=max_it,
                            tol=tol,
                            disp=disp,
                            stiefel_options=stiefel_opt,
                            num_restarts=num_restarts)
    
    #print the model after optimization
    print "="*20
    print "After optimization:"
    print "="*20
    print model
    print "="*100
    
    #save the model
    with open('syn_model_1d.pcl', 'wb') as f:
        pickle.dump(model, f)
    print "======================="
    print "Optimized model saved !"
    print "======================="


else:
    print "Model exists...loading ASPGP model..."
    with open(modelfilepath, 'rb') as f:
        model=pickle.load(f)
    #get the active model
    active_model=model.get_active_model()
    active_model.plot()
    plt.savefig(os.path.join(syndatadir, 'link_function.pdf'))
    plt.close()
    
    #load test data
    X_test=np.load(os.path.join(syndatadir, 'X_test.npy'))
    Y_test=np.load(os.path.join(syndatadir, 'Y_test.npy'))
    
    #Make predictions
    Y_predict, V_predict=model.predict(X_test)
    
    #get projection matrix
    W=model.kern.W
    
    #plot
    lower=np.min(Y_test)
    upper=np.max(Y_test)
    x=np.linspace(lower, upper, 100)
    y=x
    plt.plot(x, y, '--', label='$45^{\circ}$ line')
    plt.plot(Y_test, Y_predict, '.', markersize=11, label='Observations vs prediction')
    plt.xlabel('Actual observations')
    plt.ylabel('Model predictions')
    plt.legend(loc='best')
    rmse=np.sqrt((np.sum((Y_predict-Y_test)**2))/Y_test.shape[0])
    plt.title('RMSE in predictions = '+str(rmse))
    plt.savefig(os.path.join(syndatadir, 'pred_vs_obs.pdf'))