# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:56:15 2021

@author: conor
"""

import queue
import get_reading
import FYP_Naive_Bayes_V2
import add_to_db
import flask
import time
import pandas as pd
import math


app = flask.Flask(__name__)

daytimelikelihoods = FYP_Naive_Bayes_V2.naive_bayes_impl_daytime()

# nighttimelikelihoods = FYP_Naive_Bayes_V2.naive_bayes_impl_nighttime()


@app.route('/')
def hello_world():
    return 'Hello, World!'


class MessageAnnouncer:

    def __init__(self):
        self.listeners = []

    def listen(self):
        self.listeners.append(queue.Queue(maxsize=5))
        return self.listeners[-1]

    def announce(self, msg):
        # We go in reverse order because we might have to delete an element, which will shift the
        # indices backward
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except queue.Full:
                del self.listeners[i]


announcer = MessageAnnouncer()


def format_sse(data: str, event=None) -> str:
    """Formats a string and an event name in order to follow the event stream convention.
    >>> format_sse(data=json.dumps({'abc': 123}), event='Jackson 5')
    'event: Jackson 5\\ndata: {"abc": 123}\\n\\n'
    """
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg


@app.route('/ping')
def ping():
    
    initial_id = 1
    
    while True:
    
        gen = get_reading.getReading(initial_id)
        
        reading = list(gen)
        
        reading = reading[0]
        
        TV_sensor = max(reading[4])
    
        Kitchen_motion_sensor = max(reading[5])
    
        Corridor_motion_sensor = max(reading[6])
    
        Bedroom_pressure_sensor = max(reading[7])
        
        likelihoods = []
        
        if max(reading[9]) == 0:
            likelihoods = daytimelikelihoods
            
        else:
            likelihoods = daytimelikelihoods
            
        prediction = get_prediction(likelihoods,reading)
        
        print('-------------------------')
        print('TV Sensor: ',TV_sensor, ' Corridor: ',Corridor_motion_sensor,
              ' Kitchen Sensor: ',Kitchen_motion_sensor,' Bedroom Sensor: ',Bedroom_pressure_sensor)
        print('\nPrediction: ',prediction)
        print('-------------------------')
        
        add_to_db.add_to_db(reading,prediction)
        
        initial_id = initial_id+1
        
        time.sleep(1)
        
    msg = format_sse(data=reading)
    announcer.announce(msg=msg)
    return {}, 200

def get_prediction(liklihoods,data):
    
    TV_sensor = max(data[4])
    
    Kitchen_motion_sensor = max(data[5])
    
    Corridor_motion_sensor = max(data[6])
    
    Bedroom_pressure_sensor = max(data[7])
    
    priors = liklihoods[0]
    
    tv_like = liklihoods[1]
    
    kitchen_like = liklihoods[2]
    
    corridor_like = liklihoods[3]
    
    bed_like = liklihoods[4]
    
    logLikelihood_normal = 0
    logLikelihood_abnormal = 0
    
    if TV_sensor==1:
        logLikelihood_normal = logLikelihood_normal + math.log(tv_like[0])
        logLikelihood_abnormal = logLikelihood_abnormal + math.log(tv_like[2])
        print('TV is on')
        print(logLikelihood_normal)
        print(logLikelihood_abnormal)
    elif TV_sensor==0:
        logLikelihood_normal = logLikelihood_normal + math.log(tv_like[1])
        logLikelihood_abnormal = logLikelihood_abnormal + math.log(tv_like[3])
        print('TV is off')
        print(logLikelihood_normal)
        print(logLikelihood_abnormal)
        
    if Corridor_motion_sensor==1:
        logLikelihood_normal = logLikelihood_normal + math.log(kitchen_like[0])
        logLikelihood_abnormal = logLikelihood_abnormal + math.log(kitchen_like[2])
        print('Corridor is on')
        print(logLikelihood_normal)
        print(logLikelihood_abnormal)
        
    elif Corridor_motion_sensor==0:
        logLikelihood_normal = logLikelihood_normal + math.log(kitchen_like[1])
        logLikelihood_abnormal = logLikelihood_abnormal + math.log(kitchen_like[3])
        print('Corridor is off')
        print(logLikelihood_normal)
        print(logLikelihood_abnormal)
    
    if Kitchen_motion_sensor==1:
        logLikelihood_normal = logLikelihood_normal + math.log(corridor_like[0])
        logLikelihood_abnormal = logLikelihood_abnormal + math.log(corridor_like[2])
        print('kitchen is on')
        print(logLikelihood_normal)
        print(logLikelihood_abnormal)
        
    elif Kitchen_motion_sensor==0:
        logLikelihood_normal = logLikelihood_normal + math.log(corridor_like[1])
        logLikelihood_abnormal = logLikelihood_abnormal + math.log(corridor_like[3])
        print('kitchen is off')
        print(logLikelihood_normal)
        print(logLikelihood_abnormal)
        
    if Bedroom_pressure_sensor==1:
        logLikelihood_normal = logLikelihood_normal + math.log(bed_like[0])
        logLikelihood_abnormal = logLikelihood_abnormal + math.log(bed_like[2])
        print('bedroom is on')
        print(logLikelihood_normal)
        print(logLikelihood_abnormal)
        
    elif Bedroom_pressure_sensor==0:
        logLikelihood_normal = logLikelihood_normal + math.log(bed_like[1])
        logLikelihood_abnormal = logLikelihood_abnormal + math.log(bed_like[3])
        print('bedroom is off')
        print(logLikelihood_normal)
        print(logLikelihood_abnormal)
        
    prediction = []
        
    if logLikelihood_normal - logLikelihood_abnormal > math.log(priors[0]) - math.log(priors[1]):
        print(logLikelihood_normal - logLikelihood_abnormal)
        print(math.log(priors[0]) - math.log(priors[1]))
        prediction.append(0)
    else:
        print(logLikelihood_normal - logLikelihood_abnormal)
        print(math.log(priors[0]) - math.log(priors[1]))
        prediction.append(1)
                            
    return prediction
    
    
    # else:
    #     print('Train Model')
    #     return ('No model here')


@app.route('/listen', methods=['GET'])
def listen():

    def stream():
        messages = announcer.listen()  # returns a queue.Queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            time.sleep(1)
            yield msg

    return flask.Response(stream(), mimetype='text/event-stream')