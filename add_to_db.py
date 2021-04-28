# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:20:53 2021

@author: conor
"""
from datetime import datetime
from influxdb import InfluxDBClient

def add_to_db(data,prediction):

    client = InfluxDBClient(host='localhost', port=8086)
    
    client.switch_database('predictions')
        
    json_body = [{
        "measurement": "Multiple_sensor_readings",
        "tags": {
            "site_name": "Nans House",
            "asset": "Water resevoir",
          },
        "time": datetime.now(),
        "fields": {
            "prediction": prediction[0],
            "reading_id": max(data[3]),
            "TV_sensor": max(data[4]),
            "Kitchen_motion_sensor": max(data[5]),
            "Corridor_motion_sensor": max(data[6]),
            "Bedroom_pressure_sensor": max(data[7]),
            # "Bathroom_motion_sensor": max(data[4]),
            "daytime": max(data[9]),
        }
    }]
        
    client.write_points(json_body)
        
    reading = max(data[3])
               
    bind_params={"reading_id": int(reading)}
        
    results = client.query(query='SELECT * FROM "Multiple_sensor_readings" WHERE ("reading_id" =  $reading_id )', bind_params=bind_params)
    
    print('\n Prediction added to prediction db')
    # print(results.raw)
        
    # points = results.get_points("Multiple_sensor_readings")
        
    # for point in points:
    #     print('\n Prediction added to prediction db -------')
    #     print(point)
        
if __name__ == "__main__":
    add_to_db()