# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:08:48 2021

@author: conor
"""

import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np

path = 'D:\datasets'
list_of_files = []

for root, dirs, files in os.walk(path):
    for file in files:
        	list_of_files.append(os.path.join(root,file))
            
sensor_list = pd.DataFrame()

dataset = pd.read_csv('D:\datasets')
    
print(dataset)
            
# for name in list_of_files:
#     dataset = pd.read_csv(name)
    
#     print(dataset.columns)
    
#     sensor_set = dataset['sensor_id']==5894
    
#     dataset = dataset[sensor_set]
    
#     sensor_list.add(dataset)
    
# for row in sensor_list():
#     print(row)