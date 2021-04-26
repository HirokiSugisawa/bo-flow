import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from pyDOE import *  # Library for Latin Hypercube Sampling (LHS)
warnings.filterwarnings('ignore')

def change_to_discrete(X, grids):
    ndata = len(grids)
    minv = np.inf
    for igrid in range(ndata):
        close = np.sqrt( np.sum( np.square( X - grids[igrid]) ))
        if(minv > close):
            minv = close
            minarg = igrid
            xclose = grids[igrid]
    return xclose, minarg

def avoid_clustering( aqufun, keep_arg, ntexp ):
    for imax in range(1, ntexp+1):
        maxindex = np.where(aqufun==np.sort(aqufun)[-imax])[0][0]
        flag = 1
        for ikeep in keep_arg:
            if maxindex == ikeep:
                flag = 0
                break
        if flag == 1:
            break       
    return maxindex

# Define the function
ntry = 1
def experiment(conditions, mvalue, nexp):
    global ntry 
    print('>>> Try : %d / %d'%(ntry, nexp))
    print('>>> I guess Y%% will be: %.1f %%'%(mvalue) )
    print('>>> Next condition is:')
    print('>>> Temperature(C):   %.1f'%(conditions[0]) )
    print('>>> pka:              %.1f'%(conditions[1]) )
    print('>>> Reaction time(s): %.1f'%(conditions[2]) )
    print('>>> Concentration(M): %.1f'%(conditions[3]) )
    new_yield = int( float(input(">>> Input y : ")) )
    ntry += 1  
    return new_yield


# ========================================================== 
# == Initial Conditions
# ========================================================== 
# Set the experimental conditions (10500 combinations)
# Variables           [bottom, top, interval]
# Temperature[C]:     [-20,  40,   10]
# pka:                [ 5.2, 10.0, unique]
# Reaction time[s]:   [ 0.1, 5.0,  0.1]
# Concentration[M]    [ 0.1, 0.5,  0.1]
list_conditions = [ np.arange(-20.0, 40.1, 10.0),
                    np.array([ 5.20, 7.00, 7.40, 7.70, 8.90, 10.0]),
                    np.arange( 0.10, 5.01, 0.10),
                    np.arange( 0.10, 0.51, 0.10)]

# Set Bounds: [under, upper, interval]
bounds = np.array([[-20.0, 40.0],  # Temperature
                   [  5.2, 10.0],  # pka
                   [  0.1,  5.0],  # Reaction time
                   [  0.1,  0.5]]) # Concentration
diffs = bounds[:, 1] - bounds[:, 0]

# Generate Grids
grid = np.empty((0,4), float)
for x1 in list_conditions[0]:
    for x2 in list_conditions[1]:
        for x3 in list_conditions[2]:
            for x4 in list_conditions[3]:
                grid = np.append(grid, np.array([[x1, x2, x3, x4]]), axis=0)
ngrid, ndim = grid.shape

# ========================================================== 
# == BAYESIAN OPTIMIZATION
# ========================================================== 
ninitial = 5         # Number of initial training points
nexp     = 1000      # Maximum number of experiments

# Generating training dataset
lhsampling = lhs(ndim, ninitial) * diffs + bounds[:,0]
xtrain = np.zeros((ninitial, ndim))
ytrain = np.zeros( ninitial )

keep_arg = np.array([], dtype='int')
for itrain in range(ninitial):
    print('>>> Collect initial five points:')
    xtrain[itrain], argmnt = change_to_discrete( lhsampling[itrain], grid )
    ytrain[itrain] = experiment( xtrain[itrain], 0 , ninitial )
    keep_arg = np.append(keep_arg, argmnt)
    print('================================\n')

# Identify best yield and conditions
max_arg = np.argmax( ytrain )
best_yield = ytrain[ max_arg ]
best_condition = xtrain[max_arg]
ymax = np.ones(ninitial)*np.max(ytrain)
for iexp in range(1, nexp+1):
    # Break point
    if best_yield >= 75.0:
        print()
        break

    # Gaussian Process Regression
    kernel = C(constant_value=1.0) * RBF( length_scale=np.ones(ndim) )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=30, alpha=1.E-3)
    gp.fit(X=xtrain, y=ytrain)

    # Output the kernel parameters
    ndata = len(xtrain)
    opt_data = np.zeros( (ndata, ndim+1) )
    opt_data[:,:4] = xtrain 
    opt_data[:, 4] = ytrain 

    # Construct the UCB aquisition function
    kappa = 0.1
    y_pred, sigma = gp.predict( grid, return_std=True )
    UCB = y_pred + kappa * sigma

    # Search best conditions
    ntotal = len(keep_arg)
    max_index = avoid_clustering( UCB, keep_arg, ntotal )
    keep_arg  = np.append(keep_arg, max_index)
    new_condition = grid[max_index]
    new_yield = experiment( new_condition, UCB[max_index], nexp )

    # Update experimental data
    new_yield = float( new_yield )
    new_condition = np.atleast_2d( grid[max_index] )
    ytrain = np.append(ytrain, new_yield)
    xtrain = np.append(xtrain, new_condition, axis=0)

    # Identify best yield and conditions
    max_arg = np.argmax( ytrain )
    best_yield = ytrain[max_arg]
    best_condition = xtrain[max_arg]
    ymax = np.append(ymax, best_yield)
    print('================================\n')


# Plot all the process of the Bayesian optimization
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(1,1,1)
label = 'Best cond: [%.2f  %.2f  %.2f  %.2f]\nyield: %.2f'%(best_condition[0], best_condition[1], 
                                                            best_condition[2], best_condition[3], best_yield)
axes.set_xlim(0, len(ytrain)+1)
axes.set_ylim(np.min(ytrain)-10.0, np.max(ytrain)+10.0)
axes.text(0.0, np.max(ytrain), label)
axes.set_xlabel('Number of experiments')
axes.set_ylabel('Yield [%]')
axes.plot(np.arange(1, len(ytrain)+1), ytrain, c='blue' )
axes.plot(np.arange(1, len(ymax)  +1), ymax,   c='black' )
axes.scatter(np.arange(1, len(ytrain)+1), ytrain, c='blue',  label='Sampling yield' )
axes.scatter(np.arange(1, len(ymax)  +1), ymax,   c='black', label='Max yield' )
axes.axvspan(0, 5.5, color = "coral", alpha=0.5, label='Initial data')
axes.axhspan(75, 100, color = "gray",  alpha=0.5, label='Target yields [%]')
axes.legend(loc = 'lower right')
fig.savefig( 'Bayesian_optimization.png' )