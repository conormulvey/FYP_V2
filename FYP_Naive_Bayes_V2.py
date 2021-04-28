# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:31:10 2021

@author: conor
"""

import pandas as pd
import numpy as np
import math
from sklearn import metrics


def naive_bayes():
    normal_dataset = pd.read_excel('D:/datasets/hourly/Combined_Sensor_Dataset_Test_Daytime.xlsx',header=0)
    
    data = normal_dataset[['TV_sensor','Kitchen_motion_sensor','Corridor_motion_sensor','Bedroom_pressure_sensor','Bathroom_motion_sensor']]
    target = normal_dataset['anomaly']
    
    train_index = np.arange(int(1*len(target)))    
    # test_index = np.arange(int(0.8*len(target)), len(target))
    
    total = len(target[train_index])
    normal = sum(target[train_index]==0)
    abnormal = sum(target[train_index]==1)    
    print("Total number of minutes:",total)    
    print("Normal:",normal)
    print("Abnormal:",abnormal)
    
    prior_normal = normal / total
    prior_abnormal = abnormal / total
    print("Prior survivors:", prior_normal)
    print("Prior casulties:", prior_abnormal)
    
    # Sensor 3
    TV_sensor_on_normal = sum((data.iloc[train_index]["TV_sensor"]==1) & (target[train_index]==0))
    TV_sensor_off_normal= sum((data.iloc[train_index]["TV_sensor"]==0) & (target[train_index]==0))
    TV_sensor_on_abnormal = sum((data.iloc[train_index]["TV_sensor"]==1) & (target[train_index]==1))
    TV_sensor_off_abnormal = sum((data.iloc[train_index]["TV_sensor"]==0) & (target[train_index]==1))
    print("TV_sensor open normal:", TV_sensor_on_normal)
    print("TV_sensor closed normal:", TV_sensor_off_normal)
    print("TV_sensor open abnormal:", TV_sensor_on_abnormal)
    print("TV_sensor closed abnormal:", TV_sensor_off_abnormal)
    
    # Sensor 1
    Kitchen_motion_sensor_active_normal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==1) & (target[train_index]==0))
    Kitchen_motion_sensor_deactive_normal= sum((data.iloc[train_index]["Kitchen_motion_sensor"]==0) & (target[train_index]==0))
    Kitchen_motion_sensor_active_abnormal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==1) & (target[train_index]==1))
    Kitchen_motion_sensor_deactive_abnormal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==0) & (target[train_index]==1))
    print("Kitchen_motion_sensor open normal:", Kitchen_motion_sensor_active_normal)
    print("Kitchen_motion_sensor closed normal:", Kitchen_motion_sensor_deactive_normal)
    print("Kitchen_motion_sensor open abnormal:", Kitchen_motion_sensor_active_abnormal)
    print("Kitchen_motion_sensor closed abnormal:", Kitchen_motion_sensor_deactive_abnormal)
    
    Corridor_motion_active_normal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==1) & (target[train_index]==0))
    Corridor_motion_deactive_normal= sum((data.iloc[train_index]["Corridor_motion_sensor"]==0) & (target[train_index]==0))
    Corridor_motion_active_abnormal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==1) & (target[train_index]==1))
    Corridor_motion_deactive_abnormal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==0) & (target[train_index]==1))
    print("Corridor open normal:", Corridor_motion_active_normal)
    print("Corridor closed normal:", Corridor_motion_deactive_normal)
    print("Corridor open abnormal:", Corridor_motion_active_abnormal)
    print("Corridor closed abnormal:", Corridor_motion_deactive_abnormal)
    
    # Sensor 2
    bed_pressure_active_normal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==1) & (target[train_index]==0))
    bed_pressure_deactive_normal= sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==0) & (target[train_index]==0))
    bed_pressure_active_abnormal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==1) & (target[train_index]==1))
    bed_pressure_deactive_abnormal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==0) & (target[train_index]==1))
    print("In Bed normal:", bed_pressure_active_normal)
    print("Out Bed normal:", bed_pressure_deactive_normal)
    print("In Bed abnormal:", bed_pressure_active_abnormal)
    print("Out Bed abnormal:", bed_pressure_deactive_abnormal)
      
    # Bathroom_motion_sensor_active_normal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==1) & (target[train_index]==0))
    # Bathroom_motion_sensor_deactive_normal= sum((data.iloc[train_index]["Bathroom_motion_sensor"]==0) & (target[train_index]==0))
    # Bathroom_motion_sensor_active_abnormal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==1) & (target[train_index]==1))
    # Bathroom_motion_sensor_deactive_abnormal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==0) & (target[train_index]==1))
    # print("Front door open normal:", Bathroom_motion_sensor_active_normal)
    # print("Front door closed normal:", Bathroom_motion_sensor_deactive_normal)
    # print("Front door open abnormal:", Bathroom_motion_sensor_active_abnormal)
    # print("Front door closed abnormal:", Bathroom_motion_sensor_deactive_abnormal)
    
    alpha = 1
    
    # Sensor TV
    likelihood_TV_sensor_on_normal = (TV_sensor_on_normal + alpha) / (normal + 2*alpha)
    likelihood_TV_sensor_off_normal = (TV_sensor_off_normal + alpha) / (normal + 2*alpha)
    print("Likelihood in TV/normal:", likelihood_TV_sensor_on_normal)
    print("Likelihood out TV/normal:", likelihood_TV_sensor_off_normal)

    likelihood_TV_on_abnormal = (TV_sensor_on_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_TV_off_abnormal = (TV_sensor_off_abnormal + alpha) / (abnormal + 2*alpha)
    print("Likelihood in TV/abnormal:", likelihood_TV_on_abnormal)
    print("Likelihood out TV/abnormal:", likelihood_TV_off_abnormal)
    
    # Sensor Kitchen
    likelihood_Kitchen_motion_active_normal = (Kitchen_motion_sensor_active_normal + alpha) / (normal + 2*alpha)
    likelihood_Kitchen_motion_deactive_normal = (Kitchen_motion_sensor_deactive_normal + alpha) / (normal + 2*alpha)   
    print("Likelihood Kitchen_motion_sensor active/normal:", likelihood_Kitchen_motion_active_normal)
    print("Likelihood Kitchen_motion_sensor deactive/normal:", likelihood_Kitchen_motion_deactive_normal)

    likelihood_Kitchen_motion_active_abnormal = (Kitchen_motion_sensor_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_Kitchen_motion_deactive_abnormal = (Kitchen_motion_sensor_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    print("Likelihood Kitchen_motion_sensor active/abnormal:", likelihood_Kitchen_motion_active_abnormal)
    print("Likelihood Kitchen_motion_sensor deactive/abnormal:", likelihood_Kitchen_motion_deactive_abnormal)

    
    # Sensor Corridor
    likelihood_Corridor_motion_active_normal = (Corridor_motion_active_normal + alpha) / (normal + 2*alpha)
    likelihood_Corridor_motion_deactive_normal = (Corridor_motion_deactive_normal + alpha) / (normal + 2*alpha)
    print("Likelihood Corridor active/normal:", likelihood_Corridor_motion_active_normal)
    print("Likelihood Corridor deactive/normal:", likelihood_Corridor_motion_deactive_normal)

    likelihood_Corridor_motion_active_abnormal = (Corridor_motion_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_Corridor_motion_deactive_abnormal = (Corridor_motion_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    print("Likelihood Corridor active/abnormal:", likelihood_Corridor_motion_active_abnormal)
    print("Likelihood Corridor deactive/abnormal:", likelihood_Corridor_motion_deactive_abnormal)
    
    # Sensor Bed
    likelihood_bed_pressure_active_normal = (bed_pressure_active_normal + alpha) / (normal + 2*alpha)
    likelihood_pressure_deactive__normal = (bed_pressure_deactive_normal + alpha) / (normal + 2*alpha)
    print("Likelihood in bed/normal:", likelihood_bed_pressure_active_normal)
    print("Likelihood out bed/normal:", likelihood_pressure_deactive__normal)

    likelihood_bed_pressure_active_abnormal = (bed_pressure_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_pressure_deactive_abnormal = (bed_pressure_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    print("Likelihood in bed/abnormal:", likelihood_bed_pressure_active_abnormal)
    print("Likelihood out bed/abnormal:", likelihood_pressure_deactive_abnormal)
    
    # Sensor Bath  
    # likelihood_Bathroom_motion_sensor_active_normal = (Bathroom_motion_sensor_active_normal + alpha) / (normal + 2*alpha)
    # likelihood_Bathroom_motion_sensor_deactive__normal = (Bathroom_motion_sensor_deactive_normal + alpha) / (normal + 2*alpha)
    # print("Likelihood in bed/normal:", likelihood_Bathroom_motion_sensor_active_normal)
    # print("Likelihood out bed/normal:", likelihood_Bathroom_motion_sensor_deactive__normal)

    # likelihood_Bathroom_motion_sensor_active_abnormal = (Bathroom_motion_sensor_active_abnormal + alpha) / (abnormal + 2*alpha)
    # likelihood_Bathroom_motion_sensor_deactive_abnormal = (Bathroom_motion_sensor_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    # print("Likelihood in bed/abnormal:", likelihood_Bathroom_motion_sensor_active_abnormal)
    # print("Likelihood out bed/abnormal:", likelihood_Bathroom_motion_sensor_deactive_abnormal)
    
    prediction = []
    for i in train_index:
        logLikelihood_normal = 0
        logLikelihood_abnormal = 0
        
        if data.iloc[i]["TV_sensor"]==1:
            logLikelihood_normal = logLikelihood_normal + math.log(likelihood_TV_sensor_on_normal)
            logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_TV_on_abnormal)
        elif data.iloc[i]["TV_sensor"]==0:
            logLikelihood_normal = logLikelihood_normal + math.log(likelihood_TV_sensor_off_normal)
            logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_TV_off_abnormal)
            
        if data.iloc[i]["Corridor_motion_sensor"]==1:
            logLikelihood_normal = logLikelihood_normal + math.log(likelihood_Corridor_motion_active_normal)
            logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_Corridor_motion_active_abnormal)
        elif data.iloc[i]["Corridor_motion_sensor"]==0:
            logLikelihood_normal = logLikelihood_normal + math.log(likelihood_Corridor_motion_deactive_normal)
            logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_Corridor_motion_deactive_abnormal)
        
        if data.iloc[i]["Kitchen_motion_sensor"]==1:
            logLikelihood_normal = logLikelihood_normal + math.log(likelihood_Kitchen_motion_active_normal)
            logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_Kitchen_motion_active_abnormal)
        elif data.iloc[i]["Kitchen_motion_sensor"]==0:
            logLikelihood_normal = logLikelihood_normal + math.log(likelihood_Kitchen_motion_deactive_normal)
            logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_Kitchen_motion_deactive_abnormal)
            
        if data.iloc[i]["Bedroom_pressure_sensor"]==1:
            logLikelihood_normal = logLikelihood_normal + math.log(likelihood_bed_pressure_active_normal)
            logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_bed_pressure_active_abnormal)
        elif data.iloc[i]["Bedroom_pressure_sensor"]==0:
            logLikelihood_normal = logLikelihood_normal + math.log(likelihood_pressure_deactive__normal)
            logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_pressure_deactive_abnormal)
            
        # if data.iloc[i]["Bathroom_motion_sensor"]==1:
        #     logLikelihood_normal = logLikelihood_normal + math.log(likelihood_Bathroom_motion_sensor_active_normal)
        #     logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_Bathroom_motion_sensor_active_abnormal)
        # elif data.iloc[i]["Bathroom_motion_sensor"]==0:
        #     logLikelihood_normal = logLikelihood_normal + math.log(likelihood_Bathroom_motion_sensor_active_abnormal)
        #     logLikelihood_abnormal = logLikelihood_abnormal + math.log(likelihood_Bathroom_motion_sensor_deactive_abnormal)
            
        # print(logLikelihood_normal)
        
        # print(logLikelihood_abnormal)
            
        if logLikelihood_normal - logLikelihood_abnormal > math.log(prior_normal) - math.log(prior_abnormal):
            # print(logLikelihood_normal - logLikelihood_abnormal)
            # print(math.log(prior_abnormal) - math.log(prior_normal))
            prediction.append(0)
        else:
            # print(logLikelihood_normal - logLikelihood_abnormal)
            # print(math.log(prior_abnormal) - math.log(prior_normal))
            prediction.append(1)
            
    # print(data.iloc[test_index])
    
    confusion = metrics.confusion_matrix(target[train_index], prediction)
    print(confusion)
    # print('True Positive = ',confusion[0][0])
    # print('False Positive = ',confusion[0][1])
    # print('True Negative = ',confusion[1][1])
    # print('False Negative = ',confusion[1][0])
    
    accuracy = metrics.accuracy_score(target[train_index], prediction)
    print(accuracy)
    
def naive_bayes_impl_daytime():
    # day,hour,minute,reading_id,TV_sensor,Kitchen_motion_sensor,Corridor_motion_sensor,Bedroom_pressure_sensor,Bathroom_motion_sensor
    # reading = [27, 0, 3, (287.0,), (0.0,), (0.0,), (0.0,), (0.0,), 0.0]
    
    normal_dataset = pd.read_excel('D:/datasets/hourly/Combined_Sensor_Dataset_Test_Daytime.xlsx',header=0)
    
    
    data = normal_dataset[['TV_sensor','Kitchen_motion_sensor','Corridor_motion_sensor','Bedroom_pressure_sensor','Bathroom_motion_sensor']]
    target = normal_dataset['anomaly']
    
    train_index = np.arange(int(0.8*len(target)))    
    test_index = np.arange(int(0.8*len(target)), len(target))
    
    total = len(target[train_index])
    normal = sum(target[train_index]==0)
    abnormal = sum(target[train_index]==1)    
    # print("Total number of minutes:",total)    
    # print("Normal:",normal)
    # print("Abnormal:",abnormal)
    
    prior_normal = normal / total
    prior_abnormal = abnormal / total
    # print("Prior survivors:", prior_normal)
    # print("Prior casulties:", prior_abnormal)
    priors = [prior_normal,prior_abnormal]
    
    # Sensor 3
    TV_sensor_on_normal = sum((data.iloc[train_index]["TV_sensor"]==1) & (target[train_index]==0))
    TV_sensor_off_normal= sum((data.iloc[train_index]["TV_sensor"]==0) & (target[train_index]==0))
    TV_sensor_on_abnormal = sum((data.iloc[train_index]["TV_sensor"]==1) & (target[train_index]==1))
    TV_sensor_off_abnormal = sum((data.iloc[train_index]["TV_sensor"]==0) & (target[train_index]==1))
    # print("TV_sensor open normal:", TV_sensor_on_normal)
    # print("TV_sensor closed normal:", TV_sensor_off_normal)
    # print("TV_sensor open abnormal:", TV_sensor_on_abnormal)
    # print("TV_sensor closed abnormal:", TV_sensor_off_abnormal)
    
    # Sensor 1
    Kitchen_motion_sensor_active_normal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==1) & (target[train_index]==0))
    Kitchen_motion_sensor_deactive_normal= sum((data.iloc[train_index]["Kitchen_motion_sensor"]==0) & (target[train_index]==0))
    Kitchen_motion_sensor_active_abnormal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==1) & (target[train_index]==1))
    Kitchen_motion_sensor_deactive_abnormal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==0) & (target[train_index]==1))
    # print("Kitchen_motion_sensor open normal:", Kitchen_motion_sensor_active_normal)
    # print("Kitchen_motion_sensor closed normal:", Kitchen_motion_sensor_deactive_normal)
    # print("Kitchen_motion_sensor open abnormal:", Kitchen_motion_sensor_active_abnormal)
    # print("Kitchen_motion_sensor closed abnormal:", Kitchen_motion_sensor_deactive_abnormal)
    
    Corridor_motion_active_normal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==1) & (target[train_index]==0))
    Corridor_motion_deactive_normal= sum((data.iloc[train_index]["Corridor_motion_sensor"]==0) & (target[train_index]==0))
    Corridor_motion_active_abnormal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==1) & (target[train_index]==1))
    Corridor_motion_deactive_abnormal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==0) & (target[train_index]==1))
    # print("Corridor open normal:", Corridor_motion_active_normal)
    # print("Corridor closed normal:", Corridor_motion_deactive_normal)
    # print("Corridor open abnormal:", Corridor_motion_active_abnormal)
    # print("Corridor closed abnormal:", Corridor_motion_deactive_abnormal)
    
    # Sensor 2
    bed_pressure_active_normal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==1) & (target[train_index]==0))
    bed_pressure_deactive_normal= sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==0) & (target[train_index]==0))
    bed_pressure_active_abnormal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==1) & (target[train_index]==1))
    bed_pressure_deactive_abnormal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==0) & (target[train_index]==1))
    # print("In Bed normal:", bed_pressure_active_normal)
    # print("Out Bed normal:", bed_pressure_deactive_normal)
    # print("In Bed abnormal:", bed_pressure_active_abnormal)
    # print("Out Bed abnormal:", bed_pressure_deactive_abnormal)
      
    Bathroom_motion_sensor_active_normal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==1) & (target[train_index]==0))
    Bathroom_motion_sensor_deactive_normal= sum((data.iloc[train_index]["Bathroom_motion_sensor"]==0) & (target[train_index]==0))
    Bathroom_motion_sensor_active_abnormal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==1) & (target[train_index]==1))
    Bathroom_motion_sensor_deactive_abnormal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==0) & (target[train_index]==1))
    # print("Front door open normal:", Bathroom_motion_sensor_active_normal)
    # print("Front door closed normal:", Bathroom_motion_sensor_deactive_normal)
    # print("Front door open abnormal:", Bathroom_motion_sensor_active_abnormal)
    # print("Front door closed abnormal:", Bathroom_motion_sensor_deactive_abnormal)
    
    alpha = 1
    
    # Sensor TV
    likelihood_TV_sensor_on_normal = (TV_sensor_on_normal + alpha) / (normal + 2*alpha)
    likelihood_TV_sensor_off_normal = (TV_sensor_off_normal + alpha) / (normal + 2*alpha)
    print("Likelihood in TV/normal:", likelihood_TV_sensor_on_normal)
    print("Likelihood out TV/normal:", likelihood_TV_sensor_off_normal)

    likelihood_TV_on_abnormal = (TV_sensor_on_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_TV_off_abnormal = (TV_sensor_off_abnormal + alpha) / (abnormal + 2*alpha)
    print("Likelihood in TV/abnormal:", likelihood_TV_on_abnormal)
    print("Likelihood out TV/abnormal:", likelihood_TV_off_abnormal)
    
    tv_like = [likelihood_TV_sensor_on_normal,likelihood_TV_sensor_off_normal,likelihood_TV_on_abnormal,likelihood_TV_off_abnormal]
    
    # Sensor Kitchen
    likelihood_Kitchen_motion_active_normal = (Kitchen_motion_sensor_active_normal + alpha) / (normal + 2*alpha)
    likelihood_Kitchen_motion_deactive_normal = (Kitchen_motion_sensor_deactive_normal + alpha) / (normal + 2*alpha)   
    print("Likelihood Kitchen_motion_sensor active/normal:", likelihood_Kitchen_motion_active_normal)
    print("Likelihood Kitchen_motion_sensor deactive/normal:", likelihood_Kitchen_motion_deactive_normal)

    likelihood_Kitchen_motion_active_abnormal = (Kitchen_motion_sensor_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_Kitchen_motion_deactive_abnormal = (Kitchen_motion_sensor_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    print("Likelihood Kitchen_motion_sensor active/abnormal:", likelihood_Kitchen_motion_active_abnormal)
    print("Likelihood Kitchen_motion_sensor deactive/abnormal:", likelihood_Kitchen_motion_deactive_abnormal)
    
    kitchen_like = [likelihood_Kitchen_motion_active_normal,likelihood_Kitchen_motion_deactive_normal,likelihood_Kitchen_motion_active_abnormal,likelihood_Kitchen_motion_deactive_abnormal]

    
    # Sensor Corridor
    likelihood_Corridor_motion_active_normal = (Corridor_motion_active_normal + alpha) / (normal + 2*alpha)
    likelihood_Corridor_motion_deactive_normal = (Corridor_motion_deactive_normal + alpha) / (normal + 2*alpha)
    print("Likelihood Corridor active/normal:", likelihood_Corridor_motion_active_normal)
    print("Likelihood Corridor deactive/normal:", likelihood_Corridor_motion_deactive_normal)

    likelihood_Corridor_motion_active_abnormal = (Corridor_motion_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_Corridor_motion_deactive_abnormal = (Corridor_motion_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    print("Likelihood Corridor active/abnormal:", likelihood_Corridor_motion_active_abnormal)
    print("Likelihood Corridor deactive/abnormal:", likelihood_Corridor_motion_deactive_abnormal)
    
    corridor_like = [likelihood_Corridor_motion_active_normal,likelihood_Corridor_motion_deactive_normal,likelihood_Corridor_motion_active_abnormal,likelihood_Corridor_motion_deactive_abnormal]
    
    # Sensor Bed
    likelihood_bed_pressure_active_normal = (bed_pressure_active_normal + alpha) / (normal + 2*alpha)
    likelihood_pressure_deactive__normal = (bed_pressure_deactive_normal + alpha) / (normal + 2*alpha)
    print("Likelihood in bed/normal:", likelihood_bed_pressure_active_normal)
    print("Likelihood out bed/normal:", likelihood_pressure_deactive__normal)

    likelihood_bed_pressure_active_abnormal = (bed_pressure_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_pressure_deactive_abnormal = (bed_pressure_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    print("Likelihood in bed/abnormal:", likelihood_bed_pressure_active_abnormal)
    print("Likelihood out bed/abnormal:", likelihood_pressure_deactive_abnormal)
    
    bed_like = [likelihood_bed_pressure_active_normal,likelihood_pressure_deactive__normal,likelihood_bed_pressure_active_abnormal,likelihood_pressure_deactive_abnormal]
    
    # Sensor Bath  
    # likelihood_Bathroom_motion_sensor_active_normal = (Bathroom_motion_sensor_active_normal + alpha) / (normal + 2*alpha)
    # likelihood_Bathroom_motion_sensor_deactive__normal = (Bathroom_motion_sensor_deactive_normal + alpha) / (normal + 2*alpha)
    # print("Likelihood in bed/normal:", likelihood_Bathroom_motion_sensor_active_normal)
    # print("Likelihood out bed/normal:", likelihood_Bathroom_motion_sensor_deactive__normal)

    # likelihood_Bathroom_motion_sensor_active_abnormal = (Bathroom_motion_sensor_active_abnormal + alpha) / (abnormal + 2*alpha)
    # likelihood_Bathroom_motion_sensor_deactive_abnormal = (Bathroom_motion_sensor_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    # print("Likelihood in bed/abnormal:", likelihood_Bathroom_motion_sensor_active_abnormal)
    # print("Likelihood out bed/abnormal:", likelihood_Bathroom_motion_sensor_deactive_abnormal)
    
    likelihoods = [priors,tv_like,kitchen_like,corridor_like,bed_like]
    
    return likelihoods
    
def naive_bayes_impl_nighttime():
    # day,hour,minute,reading_id,TV_sensor,Kitchen_motion_sensor,Corridor_motion_sensor,Bedroom_pressure_sensor,Bathroom_motion_sensor
    # reading = [27, 0, 3, (287.0,), (0.0,), (0.0,), (0.0,), (0.0,), 0.0]
    
    normal_dataset = pd.read_excel('D:/datasets/hourly/Combined_Sensor_Dataset_Test_Nighttime.xlsx',header=0)
    
    data = normal_dataset[['TV_sensor','Kitchen_motion_sensor','Corridor_motion_sensor','Bedroom_pressure_sensor','Bathroom_motion_sensor']]
    target = normal_dataset['anomaly']
    
    train_index = np.arange(int(0.8*len(target)))    
    test_index = np.arange(int(0.8*len(target)), len(target))
    
    total = len(target[train_index])
    normal = sum(target[train_index]==0)
    abnormal = sum(target[train_index]==1)    
    # print("Total number of minutes:",total)    
    # print("Normal:",normal)
    # print("Abnormal:",abnormal)
    
    prior_normal = normal / total
    prior_abnormal = abnormal / total
    # print("Prior survivors:", prior_normal)
    # print("Prior casulties:", prior_abnormal)
    priors = [prior_normal,prior_abnormal]
    
    # Sensor 3
    TV_sensor_on_normal = sum((data.iloc[train_index]["TV_sensor"]==1) & (target[train_index]==0))
    TV_sensor_off_normal= sum((data.iloc[train_index]["TV_sensor"]==0) & (target[train_index]==0))
    TV_sensor_on_abnormal = sum((data.iloc[train_index]["TV_sensor"]==1) & (target[train_index]==1))
    TV_sensor_off_abnormal = sum((data.iloc[train_index]["TV_sensor"]==0) & (target[train_index]==1))
    # print("TV_sensor open normal:", TV_sensor_on_normal)
    # print("TV_sensor closed normal:", TV_sensor_off_normal)
    # print("TV_sensor open abnormal:", TV_sensor_on_abnormal)
    # print("TV_sensor closed abnormal:", TV_sensor_off_abnormal)
    
    # Sensor 1
    Kitchen_motion_sensor_active_normal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==1) & (target[train_index]==0))
    Kitchen_motion_sensor_deactive_normal= sum((data.iloc[train_index]["Kitchen_motion_sensor"]==0) & (target[train_index]==0))
    Kitchen_motion_sensor_active_abnormal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==1) & (target[train_index]==1))
    Kitchen_motion_sensor_deactive_abnormal = sum((data.iloc[train_index]["Kitchen_motion_sensor"]==0) & (target[train_index]==1))
    # print("Kitchen_motion_sensor open normal:", Kitchen_motion_sensor_active_normal)
    # print("Kitchen_motion_sensor closed normal:", Kitchen_motion_sensor_deactive_normal)
    # print("Kitchen_motion_sensor open abnormal:", Kitchen_motion_sensor_active_abnormal)
    # print("Kitchen_motion_sensor closed abnormal:", Kitchen_motion_sensor_deactive_abnormal)
    
    Corridor_motion_active_normal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==1) & (target[train_index]==0))
    Corridor_motion_deactive_normal= sum((data.iloc[train_index]["Corridor_motion_sensor"]==0) & (target[train_index]==0))
    Corridor_motion_active_abnormal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==1) & (target[train_index]==1))
    Corridor_motion_deactive_abnormal = sum((data.iloc[train_index]["Corridor_motion_sensor"]==0) & (target[train_index]==1))
    # print("Corridor open normal:", Corridor_motion_active_normal)
    # print("Corridor closed normal:", Corridor_motion_deactive_normal)
    # print("Corridor open abnormal:", Corridor_motion_active_abnormal)
    # print("Corridor closed abnormal:", Corridor_motion_deactive_abnormal)
    
    # Sensor 2
    bed_pressure_active_normal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==1) & (target[train_index]==0))
    bed_pressure_deactive_normal= sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==0) & (target[train_index]==0))
    bed_pressure_active_abnormal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==1) & (target[train_index]==1))
    bed_pressure_deactive_abnormal = sum((data.iloc[train_index]["Bedroom_pressure_sensor"]==0) & (target[train_index]==1))
    # print("In Bed normal:", bed_pressure_active_normal)
    # print("Out Bed normal:", bed_pressure_deactive_normal)
    # print("In Bed abnormal:", bed_pressure_active_abnormal)
    # print("Out Bed abnormal:", bed_pressure_deactive_abnormal)
      
    Bathroom_motion_sensor_active_normal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==1) & (target[train_index]==0))
    Bathroom_motion_sensor_deactive_normal= sum((data.iloc[train_index]["Bathroom_motion_sensor"]==0) & (target[train_index]==0))
    Bathroom_motion_sensor_active_abnormal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==1) & (target[train_index]==1))
    Bathroom_motion_sensor_deactive_abnormal = sum((data.iloc[train_index]["Bathroom_motion_sensor"]==0) & (target[train_index]==1))
    # print("Front door open normal:", Bathroom_motion_sensor_active_normal)
    # print("Front door closed normal:", Bathroom_motion_sensor_deactive_normal)
    # print("Front door open abnormal:", Bathroom_motion_sensor_active_abnormal)
    # print("Front door closed abnormal:", Bathroom_motion_sensor_deactive_abnormal)
    
    alpha = 1
    
    # Sensor TV
    likelihood_TV_sensor_on_normal = (TV_sensor_on_normal + alpha) / (normal + 2*alpha)
    likelihood_TV_sensor_off_normal = (TV_sensor_off_normal + alpha) / (normal + 2*alpha)
    # print("Likelihood in TV/normal:", likelihood_TV_sensor_on_normal)
    # print("Likelihood out TV/normal:", likelihood_TV_sensor_off_normal)

    likelihood_TV_on_abnormal = (TV_sensor_on_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_TV_off_abnormal = (TV_sensor_off_abnormal + alpha) / (abnormal + 2*alpha)
    # print("Likelihood in TV/abnormal:", likelihood_TV_on_abnormal)
    # print("Likelihood out TV/abnormal:", likelihood_TV_off_abnormal)
    
    tv_like = [likelihood_TV_sensor_on_normal,likelihood_TV_sensor_off_normal,likelihood_TV_on_abnormal,likelihood_TV_off_abnormal]
    
    # Sensor Kitchen
    likelihood_Kitchen_motion_active_normal = (Kitchen_motion_sensor_active_normal + alpha) / (normal + 2*alpha)
    likelihood_Kitchen_motion_deactive_normal = (Kitchen_motion_sensor_deactive_normal + alpha) / (normal + 2*alpha)   
    # print("Likelihood Kitchen_motion_sensor active/normal:", likelihood_Kitchen_motion_active_normal)
    # print("Likelihood Kitchen_motion_sensor deactive/normal:", likelihood_Kitchen_motion_deactive_normal)

    likelihood_Kitchen_motion_active_abnormal = (Kitchen_motion_sensor_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_Kitchen_motion_deactive_abnormal = (Kitchen_motion_sensor_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    # print("Likelihood Kitchen_motion_sensor active/abnormal:", likelihood_Kitchen_motion_active_abnormal)
    # print("Likelihood Kitchen_motion_sensor deactive/abnormal:", likelihood_Kitchen_motion_deactive_abnormal)
    
    kitchen_like = [likelihood_Kitchen_motion_active_normal,likelihood_Kitchen_motion_deactive_normal,likelihood_Kitchen_motion_active_abnormal,likelihood_Kitchen_motion_deactive_abnormal]

    
    # Sensor Corridor
    likelihood_Corridor_motion_active_normal = (Corridor_motion_active_normal + alpha) / (normal + 2*alpha)
    likelihood_Corridor_motion_deactive_normal = (Corridor_motion_deactive_normal + alpha) / (normal + 2*alpha)
    # print("Likelihood Corridor active/normal:", likelihood_Corridor_motion_active_normal)
    # print("Likelihood Corridor deactive/normal:", likelihood_Corridor_motion_deactive_normal)

    likelihood_Corridor_motion_active_abnormal = (Corridor_motion_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_Corridor_motion_deactive_abnormal = (Corridor_motion_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    # print("Likelihood Corridor active/abnormal:", likelihood_Corridor_motion_active_abnormal)
    # print("Likelihood Corridor deactive/abnormal:", likelihood_Corridor_motion_deactive_abnormal)
    
    corridor_like = [likelihood_Corridor_motion_active_normal,likelihood_Corridor_motion_deactive_normal,likelihood_Corridor_motion_active_abnormal,likelihood_Corridor_motion_deactive_abnormal]
    
    # Sensor Bed
    likelihood_bed_pressure_active_normal = (bed_pressure_active_normal + alpha) / (normal + 2*alpha)
    likelihood_pressure_deactive__normal = (bed_pressure_deactive_normal + alpha) / (normal + 2*alpha)
    # print("Likelihood in bed/normal:", likelihood_bed_pressure_active_normal)
    # print("Likelihood out bed/normal:", likelihood_pressure_deactive__normal)

    likelihood_bed_pressure_active_abnormal = (bed_pressure_active_abnormal + alpha) / (abnormal + 2*alpha)
    likelihood_pressure_deactive_abnormal = (bed_pressure_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    # print("Likelihood in bed/abnormal:", likelihood_bed_pressure_active_abnormal)
    # print("Likelihood out bed/abnormal:", likelihood_pressure_deactive_abnormal)
    
    bed_like = [likelihood_bed_pressure_active_normal,likelihood_pressure_deactive__normal,likelihood_bed_pressure_active_abnormal,likelihood_pressure_deactive_abnormal]
    
    # Sensor Bath  
    # likelihood_Bathroom_motion_sensor_active_normal = (Bathroom_motion_sensor_active_normal + alpha) / (normal + 2*alpha)
    # likelihood_Bathroom_motion_sensor_deactive__normal = (Bathroom_motion_sensor_deactive_normal + alpha) / (normal + 2*alpha)
    # print("Likelihood in bed/normal:", likelihood_Bathroom_motion_sensor_active_normal)
    # print("Likelihood out bed/normal:", likelihood_Bathroom_motion_sensor_deactive__normal)

    # likelihood_Bathroom_motion_sensor_active_abnormal = (Bathroom_motion_sensor_active_abnormal + alpha) / (abnormal + 2*alpha)
    # likelihood_Bathroom_motion_sensor_deactive_abnormal = (Bathroom_motion_sensor_deactive_abnormal + alpha) / (abnormal + 2*alpha)
    # print("Likelihood in bed/abnormal:", likelihood_Bathroom_motion_sensor_active_abnormal)
    # print("Likelihood out bed/abnormal:", likelihood_Bathroom_motion_sensor_deactive_abnormal)
    
    likelihoods = [priors,tv_like,kitchen_like,corridor_like,bed_like]
    
    return likelihoods

# naive_bayes_impl(0)
    
naive_bayes()
