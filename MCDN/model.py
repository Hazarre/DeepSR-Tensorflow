#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:36:28 2021
@author: samyang
"""

from tensorflow.keras import layers, regularizers, models

def activation(name, if_prelu=True):
    if if_prelu:
        return layers.PReLU(shared_axes=[1,2], name=name)
    else:
        return layers.ReLU(name=name)
    
def MCDN(M=3, D=5, L=4, C=64, l2=1e-7, 
         if_train=True, if_prelu=True, scale=4, name="MCDN"):
    model_name = name
    regularizer = regularizers.L2(l2=l2) if if_train else None
    actv = "PReLU-" if if_prelu else "ReLU-"
    
    if scale == 2:
        deconv_list = ["A"]
    elif scale == 4:
        deconv_list = ["A", "B"]
    else:
        raise Exception("Scale: 2 / 4.")
    
    # List of layers
    main = dict()
    
    main["Add"] = layers.Add(name="Add")
    main["Concat"] = layers.Concatenate(name="Concat")
    
    for i in range(4):
        filters = 1 if i + 1 == 4 else C
        kernel_size = 1 if i + 1 in [2, 4] else 3
        
        name = f"Conv-{i+1}"
        main[name] = layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size, padding="SAME",
                                   kernel_regularizer=regularizer,
                                   name=name)
        
        name = actv+f"{i+1}"
        main[name] = activation(name, if_prelu=if_prelu)
    
    
    for a in deconv_list:
        name = f"Deconv-{a}"
        main[name] = layers.Conv2DTranspose(filters=C, kernel_size=2,
                                            strides=2, padding="SAME",
                                            kernel_regularizer=regularizer,
                                            name=name)
        
        name = actv+f"{a}"
        main[name] = activation(name, if_prelu=if_prelu)
        
    # Dense Blocks
    for d in range(D):
        for l in range(L):
            filters = 2 ** (d + 1) if l + 1 != L else C
            name = f"Conv-D{d+1}.{l+1}"
            main[name] = layers.Conv2D(filters=filters, 
                                       kernel_size=3, padding="SAME", 
                                       kernel_regularizer=regularizer, 
                                       name=name)
            
            name = actv+f"D{d+1}.{l+1}"
            main[name] = activation(name, if_prelu=if_prelu)
            
            if l + 1 != L:
                name = f"Concat-D{d+1}.{l+1}"
                main[name] = layers.Concatenate(name=name)
            else:
                name = f"Add-D{d+1}"
                main[name] = layers.Add(name=name)
                
    # Forward Propagate
    
    inputs = layers.Input(shape=[None,None,1], name="Input")
    outputs = main["Conv-1"](inputs)
    outputs = main[actv+"1"](outputs)
    outputs_reserve = outputs
    
    ## MCD
    
    for m in range(M):
        outputs_temp_list = list()
        
        for d in range(D):
            outputs_skip_in_denseblock = outputs
            outputs_temp = outputs
            for l in range(L):
                outputs_dense = outputs_temp
                outputs_temp = main[f"Conv-D{d+1}.{l+1}"](outputs_temp)
                outputs_temp = main[actv+f"D{d+1}.{l+1}"](outputs_temp)
                
                if l + 1 != L:
                    merge = [outputs_temp, outputs_dense]
                    outputs_temp = main[f"Concat-D{d+1}.{l+1}"](merge)
                else:
                    merge = [outputs_temp, outputs_skip_in_denseblock]
                    outputs_temp = main[f"Add-D{d+1}"](merge)
                    outputs_temp_list.append(outputs_temp)
        
        outputs_temp_list.append(outputs_reserve)
        outputs = main["Concat"](outputs_temp_list)
        
        for i in range(2):
            outputs = main[f"Conv-{i+2}"](outputs)
            outputs = main[actv+f"{i+2}"](outputs)
            
        outputs = main["Add"]([outputs,outputs_reserve])
    
    for a in deconv_list:
        outputs = main[f"Deconv-{a}"](outputs)
        outputs = main[actv+f"{a}"](outputs)
        
    outputs = main["Conv-4"](outputs)
    outputs = main[actv+"4"](outputs)
    return models.Model(inputs, outputs, name=model_name)