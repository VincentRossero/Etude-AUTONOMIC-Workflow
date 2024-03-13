import physio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from tools import *


raw_ecg, srate = physio.read_one_channel('EEG_27.TRC',format='micromed', channel_name='ECG1+')
raw_tco2, srate = physio.read_one_channel('EEG_27.TRC',format='micromed', channel_name='tCO2+')
raw_Pnaz, srate = physio.read_one_channel('EEG_27.TRC',format='micromed', channel_name='Pnas+')


#on coupe le signal 

duree_debut= 3800
duree_max= 30000 
frames_debut = 256 * duree_debut
frames_max = 256 * duree_max

raw_tco2 = raw_tco2[frames_debut:frames_max]
raw_ecg = raw_ecg[frames_debut:frames_max]
raw_Pnaz=raw_Pnaz[frames_debut:frames_max]


raw_tco2 = abs(raw_tco2)
raw_Pnaz *=-1
time = np.arange(0, raw_Pnaz.size/srate, 1 /srate) 
smooth_Tcco2 = iirfilt(raw_tco2,srate,highcut=1) 



#permet d'améliorer la detection lorsque la baseline a tedance a remonter pendant la pause expiratoire 
parameters = physio.get_respiration_parameters('human_airflow') 
parameters['cycle_detection']['inspiration_adjust_on_derivative'] = True

resp, resp_cycles = physio.compute_respiration(raw_Pnaz, srate, parameters=parameters)
inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values


Pnaz_iirfilt = iirfilt(raw_Pnaz, srate, lowcut = 0.05, highcut = 1.5, show = False)

height = returnheight (Pnaz_iirfilt)


datascipy = frequenceScipy (Pnaz_iirfilt,srate,height,2)


respi_ind_scipy = datascipy['inds'].values
datascipy['TCO2']= smooth_Tcco2[respi_ind_scipy]

stats_quantitative (datascipy,'TCO2','freq','Valeur de la TCO2','Fréquence respiratoire (en Hz)')
plt.show ()



fig, axs = plt.subplots(nrows = 3, sharex = True)
fig.suptitle('patient 002-08 , sommeil')

ax = axs[0]
ax.plot(time, raw_tco2)
ax.set_title('tcco2')
ax.plot(time,smooth_Tcco2)
ax.set_title('tcco2')

ax = axs[1]
ax.plot(time, raw_Pnaz)
ax.plot(time,Pnaz_iirfilt)
ax.set_title('Pnaz')
ax.scatter(time[respi_ind_scipy], resp[respi_ind_scipy], marker='o', color='magenta')

ax = axs[2]
ax.plot(time, raw_Pnaz)
ax.plot(time,resp)
ax.set_title('Pnaz')
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')

ax.set_ylabel('resp NAZ')


plt.show ()

fig,axs =plt.subplots (nrows=2,sharex= True) 
ax = axs[0]
ax.plot(datascipy['inds']/srate,sliding_mean(datascipy['freq'],5))
ax.set_title('frequence respiratoire')
ax = axs[1]
ax.plot(time,smooth_Tcco2)
ax.set_title('tcco2')
plt.show ()

fig,axs  = plt.subplots (nrows=2,sharex=True)
ax = axs[0]
ax.plot(resp_cycles['inspi_time'],sliding_mean(resp_cycles['cycle_freq'],5))
ax.set_title('frequence respiratoire')
ax = axs[1]
ax.plot(time,smooth_Tcco2)
ax.set_title('tcco2')
plt.show ()

resp_cycles_Tcco2= resp_cycles.copy()
resp_cycles_Tcco2['Tcco2']= smooth_Tcco2[resp_cycles['inspi_index']]

#fig,ax = plt.subplots ()
#ax.scatter(resp_cycles_Tcco2['Tcco2'],resp_cycles_Tcco2['cycle_freq'],alpha=0.6) #alpha pour la transparence
#plt.show ()


stats_quantitative (resp_cycles_Tcco2,'Tcco2','cycle_freq','Valeur de la TCO2','Fréquence respiratoire (en Hz)')
plt.show ()

stats_quantitative (resp_cycles_Tcco2,'Tcco2','inspi_amplitude','Valeur de la TCO2','Amplitude inspiratoire')
plt.show ()



fig,ax = plt.subplots ()
fig.suptitle('Frequence de la valeur de TCCO2')
ax.hist (smooth_Tcco2,bins=np.arange(35,65,0.5))
ax.set_ylabel('Nombre d occurences ')
ax.set_xlabel('Valeur de TCCO2')
plt.show () 
