# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 07:14:26 2021

@author: conor
"""
from influxdb import InfluxDBClient
import random
import sys 
import datetime
import pandas as pd

def getReading(reading):
    
    # json_body = []
    
    #Connect to local influxdb
    #test_sensor_data currently holds 11 entries
    client = InfluxDBClient(host='localhost', port=8086, database='demonstration')
    
    # client = InfluxDBClient(host='localhost', port=8086, database='demonstration')
    
    bind_params={"reading_id": int(reading)}
    
    #retrieve all the data from the database
    #returns a json with all 11 entries in it
    # results = client.query('SELECT * FROM "Multiple_sensor_readings"')
    
    # print(results.raw)
    
    results = client.query(query='SELECT * FROM "Multiple_sensor_readings" WHERE ("reading_id" =  $reading_id )', bind_params=bind_params)
    
    # For loop retirieves values relevant to the prediction from results json
    for i in range(len(results.raw['series'][0]['values'])): 
        
        # print(results.raw['series'][0]['values'][i])

        timestamp = results.raw['series'][0]['values'][i][0]
        
        reading_id = results.raw['series'][0]['values'][i][7],
        
        daytime = results.raw['series'][0]['values'][i][7],
        
        TV_sensor = results.raw['series'][0]['values'][i][5],
        Kitchen_motion_sensor = results.raw['series'][0]['values'][i][4],
        Corridor_motion_sensor = results.raw['series'][0]['values'][i][3],
        Bedroom_pressure_sensor = results.raw['series'][0]['values'][i][2],
        Bathroom_motion_sensor = results.raw['series'][0]['values'][i][1]
        
        try:
            month   = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').month
            day     = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').day
            hour    = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').hour
            minute  = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').minute
            
        except:
            month   =  datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').month
            day     =  datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').day
            hour    =  datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').hour
            minute  =  datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').minute
            
        
        # sensor_id = results.raw['series'][0]['values'][i][2]
    
        # json_body = {"sensor_id": sensor_id 
        #                   ,"value": value
        #                   ,"month": month
        #                   ,"day": day
        #                   ,"hour": hour
        #                   ,"minute": minute
        #                   }
        
        my_list = [day,hour,minute,reading_id,TV_sensor,Kitchen_motion_sensor,Corridor_motion_sensor,Bedroom_pressure_sensor,Bathroom_motion_sensor,daytime]
        
        
        yield my_list

if __name__ == "__main__":
    getReading(9)