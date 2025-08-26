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

def calc_jmi_kraskov_XX(a,b):
    '''
    Estimate multi-dimensional mutual information from micro to micro
    Input:  a: multi-dimensional source
            b: 2d target
    '''
    # start JVM
    start_JVM()
    # 1. Construct the calculator:
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    # 2. Set any properties to non-default values:
    calc.setProperty("TIME_DIFF", "1")
    # 3. Initialise the calculator for (re-)use:
    calc.initialise(np.shape(a)[1], 2)
    # 4. Supply the sample data:
    calc.setObservations(JArray(JDouble, 2)(a), JArray(JDouble, 2)(b))
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()
    return result

def detect_syn_pairs(i, j, k, X_t):
    '''
    Compute the quantity B for certain sources and targets
    Input:  i,j: index of the sources
            k: index of the target
            X_t: all spins
    '''
    # source spins
    arr1 = np.array([np.cos(X_t[:,i]), np.sin(X_t[:,i])]).T
    arr2 = np.array([np.cos(X_t[:,j]), np.sin(X_t[:,j])]).T
    # target spin
    arr_T = np.array([np.cos(X_t[:,k]), np.sin(X_t[:,k])]).T
    # combine two spins
    comb = np.hstack((arr1,arr2))

    summation = calc_jmi_kraskov_XX(arr1,arr_T) + calc_jmi_kraskov_XX(arr2,arr_T)
    jmi = calc_jmi_kraskov_XX(comb, arr_T)
    return jmi - summation

def detect_syn_single(k, X_t):
    '''
    Compute all pairwise mutual information about a fixed k
    Input:  k: index of the target
            X_t: all spins
    '''
    n = np.shape(X_t)[1]
    results = Parallel(n_jobs=16)(delayed(detect_syn_pairs)(i, j, k, X_t) for i in range(n) for j in range(i+1, n))
    return results

def detect_syn_all(X_t):
    '''
    Compute all pairwise mutual information for all k
    Input:  X_t: all spins
    '''
    n = np.shape(X_t)[1]
    results = []
    for k in range(n):
        results.append(detect_syn_single(k, X_t))
    return results

