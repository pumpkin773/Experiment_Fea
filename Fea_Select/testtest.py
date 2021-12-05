#!/usr/bin/python
# -*- coding:utf-8 -*-


import eventlet
import time

eventlet.monkey_patch()

time_limit = 3  # set timeout time

with eventlet.Timeout(time_limit, False):
    time.sleep(5)
    cal = 9*9
    print('run over')
print(cal)
print('over')

