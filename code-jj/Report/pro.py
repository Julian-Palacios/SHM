from Python_tools.core import Event
import pandas as pd

event = Event.load_event_properties('D:/SHM/code-jj/Report/Event_properties.sav')

# print(event.max_station["Max_pga"].iloc[0])
print(event.station)