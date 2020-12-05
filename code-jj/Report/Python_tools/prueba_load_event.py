# -*- coding: utf-8 -*-
"""
Created on December 2020

@author: Joseph Jaramillo
"""


from waves import Event

x = Event.load_event_properties('D:/SHM/code-jj/Tools/Event_properties.sav')
print(x.epicenter)
print(x)

print(x.station)