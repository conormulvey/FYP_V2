# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:57:16 2021

@author: conor
"""

import time

import requests


if __name__ == '__main__':

    while True:
        requests.get('http://localhost:5000/ping')
        time.sleep(10)