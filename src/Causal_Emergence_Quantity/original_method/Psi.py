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

def calc_mi_kraskov(a,b):
    '''
    Estimate mutual information
    Input:  a: 1d source
            b: 1d target
    '''
    # start JVM
    start_JVM()
    # 1. Construct the calculator:
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    # 2. Set any properties to non-default values:
    calc.setProperty("TIME_DIFF", "1") # time difference
    # 3. Initialise the calculator for (re-)use:
    calc.initialise()
    # 4. Supply the sample data:
    calc.setObservations(JArray(JDouble, 1)(a), JArray(JDouble, 1)(b)) # both source and target are 1d
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()
    return result

def calc_jmi_kraskov(a,b):
    '''
    Estimate multi-dimensional mutual information
    Input:  a: source, the angle
            b: target, the energy of vortices
    '''
    # start JVM
    start_JVM()
    # 1. Construct the calculator:
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    # 2. Set any properties to non-default values:
    calc.setProperty("TIME_DIFF", "1") # time difference
    # 3. Initialise the calculator for (re-)use:
    calc.initialise(np.shape(a)[1], 1)
    # 4. Supply the sample data:
    calc.setObservations(JArray(JDouble, 2)(a), JArray(JDouble, 1)(b)) # source is 2d
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()
    return result

def micro_mi(X, V_t):
    '''
    Compute the mutual information from a single spin to vortices
    Input:  X: the angle of a spin
            V_t: energy of vortices
    '''
    X_2d = np.array([np.cos(X), np.sin(X)]).T # convert to 2d
    micro = calc_jmi_kraskov(X_2d, V_t)
    if micro<0: # mutual information is non-zero
        return 0
    return micro

def Psi(V_t, X_t):
    '''
    Compute important terms in the quantity psi
    Input:  V_t: energy of vortices
            X_t: all spins in angles
    Output: list[macro to macro, sum of micros to macro, minimum of micro to macro]
    '''
    macro = calc_mi_kraskov(V_t, V_t) # vortices to vortices
    micros = Parallel(n_jobs=16)(delayed(micro_mi)(X, V_t) for X in np.array(X_t).T) # each spin to vortices
    return [macro, sum(micros), min(micros)]


