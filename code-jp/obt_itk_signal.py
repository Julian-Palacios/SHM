import paramiko,time,os,datetime,requests
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen, urlretrieve

server="10.8.10.41"
user="root"
password="seis311"
port=22

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def nitk(url1):
    url = url1 %"select.cgi"
    f = urlopen(url)
    r = f.read()
    soup = bs(r, 'html.parser')
    itk = soup.find_all('p')
    itk2 = itk[2].getText()
    itk = itk2.split(":")
    l = len(itk)-1
    return l

def getevent(url1):
    url = url1 %"eselectt.cgi"
    f = urlopen(url)
    r = f.read()
    #
    soup = bs(r, 'html.parser')
    ev = soup.find_all("big")
    #
    even=[]
    for k in range(1,len(ev)):
        sp = ev[k].getText().split()
        #
        temp = sp[0].split("/")
        sp[0] = "%s-%s-%s" %(temp[0], temp[1], temp[2])
        #
        sp = [sp[0], sp[1], sp[4]]
        even.append(sp)
        #
##    print even
    return even

def dateS(even, fecha, fechaL):
    c = 0
    d = 0
    for e in even:
        if fecha <= e[0]:
            c += 1
            if fechaL < e[0]:#
                d += 1#
        else:
            break
##    print even[d:c]
    return even[d:c]

def getpar(url1,stn):
    url = url1 %"eselectt.cgi"
    f = urlopen(url)
    r = f.read()
    #
    soup = bs(r, 'html.parser')
    ev = soup.find_all("big")
    #
    par = []
    for k in range(1,len(ev)):
        sp = ev[k].getText().split()
        #
        p1 = sp[0].split('/')
        p2 = sp[1].split(':')
        #
        p1 = [int(x) for x in p1]
        p2 = [int(x) for x in p2]
        #
        fecha = datetime.datetime(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
        ff = fecha - datetime.timedelta(0,60)
        #
        g = "%i %02i %02i %02i %02i %02i" %(ff.year,ff.month,
                                            ff.day,ff.hour,ff.minute,ff.second)
        h = g.split()
        #
        dicc = {"year0": h[0], "month0": h[1], "day0": h[2],
                "hour0": h[3], "min0": h[4], "length0": '3',
                "stn0": stn, "pw": '800', "ph": '420', "const0": '4280',
                "unit0": 'gal', "nch0": '3', "chn0": 'N-S', "chn1": 'E-W',
                "chn2": 'U-D'}
        #
        par.append(dicc)
    return par

def req_evt(url1,i,par):
    url = url1 %"print.cgi"
    r = requests.post(url, par[i], allow_redirects=True)

def getitk(n, path, url2):
    urlW = url2 %"itk%s.txt"
    urlW = urlW %n

################################################################

url1='http://10.8.10.41/cgi-bin/%s'## Eliminar linea
url2='http://10.8.10.41/tmp/%s'## Eliminar linea
try:
    l = nitk(url1)
    even = getevent(url1)
    # print(even)
    fecha = '2020-10-06'
    fechaL = '2020-10-27'
    evS = dateS(even, fecha, fechaL)
    print(evS)
except Exception as e:
    print(e)

for j in range(len(evS)): # itera en los eventos
    base_path = './Structural-Health-Monitoring/code-jp/events'
    path='%s/%s_%s'%(base_path,evS[j][0],evS[j][0].replace(':','-'))
    if not os.path.exists(path):
        os.mkdir(path)
    print(path)

    for k in range(5): #itera en los sensores
        n = "%02i" %(k)
        stn = "itk%s" %n
        par = getpar(url1,stn)
        print(par[j])
        url = url1 %"print.cgi"
        r = requests.post(url, par[j], allow_redirects=True)
        urlW = url2 %stn+".txt"
        urlretrieve(urlW, r"%s\itk%s.txt" %(path, n))
        print("url:",url,"\nurlW:",urlW)

# carpeta="/data/eventa"
# ssh = createSSHClient(server, port, user, password)
# sftp=ssh.open_sftp()
# lista=sorted(sftp.listdir(carpeta))
# j,ini=1,lista[-1]
# print("Los 10 ultimos eventos son:")
# for i in lista[-10:]:
#     info = sftp.stat(carpeta + "/" + i)
#     print("%s) %s\t\t%.2f KB"%(j,i,info.st_size/1024.0))
#     j=j+1
# sftp.close()
# ssh.close()

# while True:
#     try:
#         ssh = createSSHClient(server, port, user, password)
#         sftp=ssh.open_sftp()
#         lista=sorted(sftp.listdir(carpeta))
#         if lista[-1]==ini:
#             continue
# ##          print("No hay nuevo evento :(")
#         else:
#             j,ini=1,lista[-1]
#             print("¡¡Nuevo Evento!!: %s"%ini.split(".")[0])
#             print("Los 10 ultimos eventos son:")
#             for i in lista[-10:]:
#                 info = sftp.stat(carpeta + "/" + i)
#                 print("%s) %s\t\t%.2f KB"%(j,i,info.st_size/1024.0))
#                 j=j+1
#         sftp.close()
#         ssh.close()
#     except Exception as e:
#         print(e)
