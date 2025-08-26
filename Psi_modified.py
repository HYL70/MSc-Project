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

def calc_jmi_kraskov(a,b):
    '''
    Estimate multi-dimensional mutual information from micro to macro
    Input:  a: multi-dimensional source
            b: 1d target
    '''
    # start JVM
    start_JVM()
    # 1. Construct the calculator:
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    # 2. Set any properties to non-default values:
    calc.setProperty("TIME_DIFF", "1")
    # 3. Initialise the calculator for (re-)use:
    calc.initialise(np.shape(a)[1], 1)
    # 4. Supply the sample data:
    calc.setObservations(JArray(JDouble, 2)(a), JArray(JDouble, 1)(b))
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()
    return result

def detect_syn_pairs(i, j, X_t, V_t):
    '''
    Compute the quantity B and MMI PID for certain sources and targets
    Input:  i,j: index of the sources
            X_t: all spins
            V_t: macro features
    Output: numpy.array([The quantity B, redundancy, synergy])
    '''
    # source spins
    arr1 = np.array([np.cos(X_t[:,i]), np.sin(X_t[:,i])]).T
    arr2 = np.array([np.cos(X_t[:,j]), np.sin(X_t[:,j])]).T
    # each contains information about vortices
    s1 = calc_jmi_kraskov(arr1,V_t) 
    s2 = calc_jmi_kraskov(arr2,V_t)
    # combined for a two-source mutual information
    comb = np.hstack((arr1,arr2))
    jmi = calc_jmi_kraskov(comb, V_t)
    return np.array([jmi-s1-s2, min(s1,s2), jmi-max(s1,s2)])

def detect_syn(X_t, V_t):
    '''
    Compute the quantity B and MMI PID for pairs of spins
    Input:  X_t: all spins
            V_t: macro features
    '''
    n = np.shape(X_t)[1]
    results = Parallel(n_jobs=16)(delayed(detect_syn_pairs)(i, j, X_t, V_t) for i in range(n) for j in range(i+1, n))
    return np.array(results)


