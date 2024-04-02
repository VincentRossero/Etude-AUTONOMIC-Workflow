import physio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from tools import *
from matplotlib.ticker import FuncFormatter

raw_thor, srate = physio.read_one_channel('EEG_137.TRC',format='micromed', channel_name='thor+')
raw_abdo, srate = physio.read_one_channel('EEG_137.TRC',format='micromed', channel_name='abdo+')
raw_ecg, srate = physio.read_one_channel('EEG_137.TRC',format='micromed', channel_name='ecg')
raw_tco2, srate = physio.read_one_channel('EEG_137.TRC',format='micromed', channel_name='tCO2+')
raw_Pnaz, srate = physio.read_one_channel('EEG_137.TRC',format='micromed', channel_name='Pnaz+')
raw_Term, srate = physio.read_one_channel('EEG_137.TRC',format='micromed', channel_name='TERM+')
raw_tco2 = abs(raw_tco2)
raw_Pnaz *=-1
threshold_high_tcO2 = 50

#troncature signaux
duree_debut= 48000
duree_max= 74000 
frames_debut = 256 * duree_debut
frames_max = 256 * duree_max
raw_tco2 = raw_tco2[frames_debut:frames_max]
raw_ecg = raw_ecg[frames_debut:frames_max]
raw_Pnaz=raw_Pnaz[frames_debut:frames_max]
raw_abdo = raw_abdo[frames_debut:frames_max]
raw_thor = raw_thor[frames_debut:frames_max]
raw_Term=raw_Term[frames_debut:frames_max]


time = np.arange(0, raw_abdo.size/srate, 1 /srate)
smooth_Tcco2 = iirfilt(raw_tco2,srate,highcut=1) 

#detection physio

parameters = physio.get_respiration_parameters('human_airflow') 
parameters['cycle_detection']['inspiration_adjust_on_derivative'] = True
resp, resp_cycles = physio.compute_respiration(raw_Pnaz, srate, parameters=parameters)
inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values

#detection scipy 

med,mad = physio.compute_median_mad(resp)
height = med + 1 * mad
datascipy = frequenceScipy (resp,srate,height,1.2)
respi_ind_scipy = datascipy['inds'].values
datascipy['TCO2']= smooth_Tcco2[respi_ind_scipy]


#detection scipy , sur thor et abdo  

thor = iirfilt(raw_thor, srate, lowcut = 0.05, highcut = 2, show = False)
meds,mads = calcul_med_mad(thor)
height = med + 1 * mad
datascipy_thor = frequenceScipy (thor,srate,height,1.2)
respi_ind_scipy_thor = datascipy_thor['inds'].values
datascipy_thor['TCO2']= smooth_Tcco2[respi_ind_scipy_thor]

abdo = iirfilt(raw_abdo, srate, lowcut = 0.05, highcut = 2, show = False)
meds, mads = calcul_med_mad(abdo)
height = med + 1 * mad
datascipy_abdo = frequenceScipy(abdo, srate, height, 1.2)
respi_ind_scipy_abdo = datascipy_abdo['inds'].values
datascipy_abdo['TCO2'] = smooth_Tcco2[respi_ind_scipy_abdo]


#correlation avec physio  

resp_cycles_Tcco2= resp_cycles.copy()
resp_cycles_Tcco2['Tcco2']= smooth_Tcco2[resp_cycles['inspi_index']]
stats_quantitative (resp_cycles_Tcco2,'Tcco2','cycle_freq','Valeur de la TCO2','Fréquence respiratoire (en Hz)')
plt.show ()

fig,axs  = plt.subplots (nrows=2,sharex=True)
ax = axs[0]
ax.plot(resp_cycles['inspi_time'],sliding_mean(resp_cycles['cycle_freq'],5))
ax.set_title('frequence respiratoire')
ax = axs[1]
ax.plot(time,smooth_Tcco2)
ax.axhline(threshold_high_tcO2, color = 'r')
ax.set_title('tcco2')
plt.show ()


stats_quantitative (datascipy,'TCO2','freq','Valeur de la TCO2','Fréquence respiratoire (en Hz)')
plt.show ()

stats_quantitative (datascipy_thor,'TCO2','freq','Valeur de la TCO2','Fréquence respiratoire (en Hz)')
plt.show ()

stats_quantitative (datascipy_abdo,'TCO2','freq','Valeur de la TCO2','Fréquence respiratoire (en Hz)')
plt.show ()

'''

fig, axs = plt.subplots(nrows = 3, sharex = True)
fig.suptitle('patient 02-003')

ax = axs[0]
ax.plot(time,resp)
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')
ax.set_title('Pnaz')

ax = axs[1]
ax.plot(time,raw_Term)
ax.set_title('Term')

ax = axs[2]
ax.plot(time,smooth_Tcco2)
ax.set_title('TCO2')
ax.axhline(threshold_high_tcO2, color = 'r')

plt.show()



fig, axs = plt.subplots(nrows = 4, sharex = True)
fig.suptitle('patient 02-003')

ax = axs[0]
ax.plot(time, raw_thor)
ax.plot(time,thor)
ax.scatter(time[respi_ind_scipy_thor], thor[respi_ind_scipy_thor], marker='o', color='magenta')
ax.set_title('thor')

ax = axs[1]
ax.plot(time,raw_abdo)
ax.plot(time,abdo)
ax.scatter(time[respi_ind_scipy_abdo], abdo[respi_ind_scipy_abdo], marker='o', color='magenta')
ax.set_title('abdo')

ax = axs[2]
ax.plot(time,raw_Pnaz)
ax.set_title('Pnaz')

ax = axs[3]
ax.plot(time,raw_Term)
ax.set_title('Term')

plt.show()

fig, axs = plt.subplots(nrows = 2, sharex = True,sharey =True)

ax = axs[0]
ax.plot(time,resp)
ax.set_title('Scipy')
ax.scatter(time[respi_ind_scipy], resp[respi_ind_scipy], marker='o', color='magenta')

ax = axs[1]
ax.plot(time,resp)
ax.set_title('Physio')
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')

plt.show ()

'''




