from XYModel2D import XYModel2D
import numpy as np
from jpype import *

def start_JVM():
    if (not isJVMStarted()):
        # Add JIDT jar library to the path
        jarLocation = "X:\\infodynamics-dist-1.6.1\\infodynamics.jar"
        # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

# all method we used

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
    calc.setProperty("TIME_DIFF", "1")
    # 3. Initialise the calculator for (re-)use:
    calc.initialise(np.shape(a)[1], 1)
    # 4. Supply the sample data:
    calc.setObservations(JArray(JDouble, 2)(a), JArray(JDouble, 1)(b))
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()
    return result

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

def calc_mi_kraskov_2d_VX(a,b):
    '''
    Estimate multi-dimensional mutual information from macro to micro
    Input:  a: source, the energy of vortices
            b: target, the angle
    '''
    # start JVM
    start_JVM()
    # 1. Construct the calculator:
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()
    # 2. Set any properties to non-default values:
    calc.setProperty("TIME_DIFF", "1")
    # 3. Initialise the calculator for (re-)use:
    calc.initialise(1, 2)
    # 4. Supply the sample data:
    calc.setObservations(JArray(JDouble, 1)(a), JArray(JDouble, 2)(b))
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()
    return float(result)

def mi_uncertainty(a, b, F, n_shuffle=100):
    '''
    Compute uncertainty/observed mutual information
    Input:  a: source
            b: target
            F: function for the estimate of mutual information
            n_shuffle: number of surrogate data sets
    '''
    observed_mi = F(a, b)

    shuffled_mis = []
    for _ in range(n_shuffle):
        b_shuffle = np.random.permutation(b)
        mi = F(a, b_shuffle)
        shuffled_mis.append(mi)

    mean_shuffled_mi = np.mean(shuffled_mis)

    return abs(mean_shuffled_mi/observed_mi)