# -*- coding: utf-8 -*-

"""
tuner_refiner

Christian Rodriguez
crodriguez0874@gmail.com
08/09/19

Summary - This function holds the tuner_refiner() function. For a parameter, the
function outputs a new list of values for RandomizedSearchCV. The new range is
based on the parameter value that yields the maximal cross validated accuracy
from the first round of tuning. The input are defined as follows:
    
parameter := the parameter value that outputted the maximal cross validated
accuracy in the first round of RandomizedSearchCV.

parameter_range := The list of values used in the first round of
RandomizedSearchCV.

integer := boolean. True if the parameter must be an integer, and False
otherwise.

This script is referred to in the model_script.py script.
"""

###############################################################################
###Loading libraries
###############################################################################

import numpy as np

###############################################################################
###Function
###############################################################################

def tuning_refiner(parameter, parameter_range, integer=True):
    
    if parameter is None:
        return([None])
        
    elif len(parameter_range) < 2:
        return('Error: parameter_range must be a list with at least 2 integer entries.')
        
    else:
        
        if parameter == parameter_range[0] and integer is False:
            
            parameter_LB = parameter_range[0]*0.1
            parameter_UB = parameter_range[0]*1.1
            parameter_steps = parameter_range[0]*0.1
            parameter_split = list(np.arange(parameter_LB,
                                             parameter_UB,
                                             parameter_steps))
            
        elif parameter == parameter_range[0] and integer is True:
            
            parameter_LB = parameter_range[0]
            parameter_UB = parameter_range[1]
            parameter_steps = (parameter_UB - parameter_LB)*0.1
            parameter_split = list(np.arange(parameter_LB,
                                             parameter_UB,
                                             parameter_steps))
            
        elif parameter == parameter_range[-1]:
            
            parameter_LB = parameter_range[-1]
            parameter_UB = parameter_range[-1]*2.1
            parameter_steps = parameter_range[-1]*0.1
            parameter_split = list(np.arange(parameter_LB,
                                             parameter_UB,
                                             parameter_steps))
            
        else:
            
            parameter_LB = parameter*0.8
            parameter_UB = parameter*1.3
            parameter_steps = parameter*0.1
            parameter_split = list(np.arange(parameter_LB,
                                             parameter_UB,
                                             parameter_steps))
            
        if integer is True:
            
            parameter_split = [int(round(x)) for x in parameter_split]
            
        else:
            
            parameter_split = [round(x, 2) for x in parameter_split]
        
        return(parameter_split)