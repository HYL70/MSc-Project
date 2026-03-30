from XYModel2D import XYModel2D
import numpy as np
from joblib import Parallel, delayed
from jpype import *
import sys

def start_JVM():
    if (not isJVMStarted()):
        # Add JIDT jar library to the path
        jarLocation = "X:\\infodynamics-dist-1.6.1\\infodynamics.jar"
        # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

def calc_mi_kraskov_2d_XX(a,b):
    '''
    Estimate multi-dimensional mutual information from micro to micro
    Input:  a: source, the angle
            b: target, the angle
    '''
    # start JVM
    start_JVM()
    # 1. Construct the calculator:
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    # 2. Set any properties to non-default values:
    calc.setProperty("TIME_DIFF", "1") # time difference
    # 3. Initialise the calculator for (re-)use:
    calc.initialise(2, 2)
    # 4. Supply the sample data:
    s = np.array([np.cos(a),np.sin(a)]).T.tolist() # convert to 2d
    d = np.array([np.cos(b),np.sin(b)]).T.tolist() # convert to 2d
    calc.setObservations(JArray(JDouble, 2)(s), JArray(JDouble, 2)(d))
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()
    if result<0: # mutual information is non-zero
        return 0
    return float(result)

def calc_mi_kraskov_2d_VX(a,b):
    '''
    Estimate multi-dimensional mutual information from macro to micro
    Input:  a: source, the energy of vortices
            b: target, the angle
    '''
    #start JVM
    start_JVM()
    # 1. Construct the calculator:
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    # 2. Set any properties to non-default values:
    # No properties were set to non-default values
    calc.setProperty("TIME_DIFF", "1")
    # 3. Initialise the calculator for (re-)use:
    calc.initialise(1, 2)
    # 4. Supply the sample data:
    d = np.array([np.cos(b),np.sin(b)]).T.tolist() # convert to 2d
    calc.setObservations(JArray(JDouble, 1)(a), JArray(JDouble, 2)(d))
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()
    if result<0: # mutual information is non-zero
        return 0
    return float(result)

def XX_list(X_t):
    '''
    Compute all pairwise mutual information from angle to angle
    Input:  X_t: all spins in angles
    '''
    n = np.shape(X_t)[1]
    results = Parallel(n_jobs=16)(delayed(calc_mi_kraskov_2d_XX)(X_t[:,i], X_t[:,j]) for i in range(n) for j in range(n))
    return results

def list_to_matrix(l):
    '''
    Convert an n by n list to a matrix
    '''
    n = int(np.sqrt(len(l)))
    return np.array(l).reshape((n,n))

def VX_list(V_t, X_t):
    '''
    Compute all mutual information from macro to each micro
    '''
    results = []
    for i in range(np.shape(X_t)[1]):
        results.append(calc_mi_kraskov_2d_VX(V_t, X_t[:,i]))
    return results

def delta_gamma(V_t, X_t):
    '''
    Compute the quantities delta and gamma
    Input:  V_t: energy of vortices
            X_t: all spins in angles
    Output: list[delta, delta with correction of redundancy, gamma]
    '''
    XXm = list_to_matrix(XX_list(X_t)) # the matrix of pairwise mutual information from angle to angle
    VXlist = VX_list(V_t, X_t) # macro to micro 
    n = np.shape(X_t)[1]
    delta = []
    delta_minus_red = []

    for j in range(np.shape(X_t)[1]):
        delta.append(VXlist[j] - sum(np.array(XXm)[:,j]))
        delta_minus_red.append(VXlist[j] - sum(np.array(XXm)[:,j]) + (n-1)*min(np.array(XXm)[:,j]))
    return max(delta), max(delta_minus_red), max(VXlist)

