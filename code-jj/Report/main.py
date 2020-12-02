from Python_tools.core import Event, Station
import os 
import pickle
import pandas as pd
import shutil

# step 1) Leer el evento del IGP y crear el objeto de la clase Event.
event = Event()
# event.load_event('D:/SHM/code-jj/Events/IGP EVENTOS/2020-0675.txt')
event.load_event('D:/SHM/code-jj/Events/IGP EVENTOS/2020-0709.sav')
# print(event.epicenter)

# step 2) leer la carpeta donde se encuentras los registros de las estaciones.
path_event = 'D:/SHM/code-jj/Events/%s' %event.event_waves_dir
with os.scandir(path_event) as f:
    s = [f.name for f in f if f.is_dir()]

# step 3) Cargar los datos de las estaciones creando los objetos de la clase Station.
stations = [Station(path_event + '/%s' %i) for i in s]

# step 4) Hacer los filtros Línea Base, Pasa Banda además de los calculos de los espectros de Fourier y de Respuesta.
for station in stations:
    station.baseLine(type='spline',order=2,dspline=1000)  
    station.passBandButterWorth(low_freq=1.0, high_freq=20.0, order=10)
    station.get_fourier(smooth_grade=25)
    station.get_sa_sd()

    # step 5) Agregar las estaciones creadas al objeto Event.
    event.add_station(station)

    # step 6) Creas las graficas de Aceleracion-Espectros de Fourier y de Respuesta.
    event.create_acc_fourier_graf(station, dpi=50, transparent=True)
    event.create_sa_sd_graf(station)

# step 7) Crear los mapas.
event.createMap01(dpi=50, transparent=True)
event.createMap02(dpi=50, transparent=True)

# strp 8) Obtener la estación cuyo PGA es máximo (Esencial para la generación del reporte).
event.get_max_station()

# step 9) Guardar las propiedades del Evento (Esencial para la generación del reporte).
event.save_event_properties('D:/SHM/code-jj/Report')

#step 10) Depurar los archivos generados automaticamente por latex en la anterior ejecución
event.purge_files(path_dir = 'D:/SHM/code-jj/Report/')

#step 11) Generar el reporte
# print("Begining Latex compilation")
# os.system('pushd d:\SHM\code-jj\Report && pdflatex Report.tex')
# os.system('pushd d:\SHM\env\Scripts && activate && pythontex --interpreter "python:py" ..\..\code-jj\Report\Report.tex')
# os.system('pushd d:\SHM\code-jj\Report && pdflatex Report.tex')
# os.system('pushd d:\SHM\env\Scripts && activate && pythontex --interpreter "python:py" ..\..\code-jj\Report\Report.tex')
# os.system('pushd d:\SHM\code-jj\Report && pdflatex Report.tex')

#step 12) Enviar el reporte para publicación
# event.send_email(report_path= "D:/SHM/code-jj/Report/Report.pdf", receivers = ["jjaramillod@uni.pe", "josdaroldplx@gmail.com", "josdarcoldplx@hotmail.com"])

print("##############")
print("#### Done ####")
print("##############")
