from waves import Event

x = Event.load_event_properties('D:/SHM/code-jj/Tools/Event_properties.sav')
print(x.epicenter)
print(x)

print(x.station)