import physio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from tools import *

raw_tco2, srate = physio.read_one_channel('EEG_18.TRC',format='micromed', channel_name='tCO2+')
raw_Spo2, srate = physio.read_one_channel('EEG_18.TRC',format='micromed', channel_name='SpO2+')

time = np.arange(0, raw_tco2.size/srate, 1 /srate) 
raw_tco2 = abs(raw_tco2)


clean_Spo2 =clean_Sao(raw_Spo2)
smooth_Tcco2 = iirfilt(raw_tco2,srate,highcut=1) 


threshold_high_tcO2 = 50
threshold_low_SpO2 = 90

above_threshold_tco2 = smooth_Tcco2 > threshold_high_tcO2
#contient un tableau indiquant un 1 à chaque passage au dessus du seuil 
diff_above_tco2 = np.diff(above_threshold_tco2.astype(int))
num_threshold_crossings_tco2 = np.sum(diff_above_tco2 == 1)

below_threshold_spo2 = clean_Spo2 < threshold_low_SpO2
diff_below_spo2 = np.diff(below_threshold_spo2.astype(int))
num_threshold_crossings_spo2 = np.sum(diff_below_spo2 == 1)

# Calcul de la durée passée au-dessus du seuil pour tCO2
durations_above_threshold_tco2 = []
index_debut_above_threshold_tco2 = []
start_index = None
for i, val in enumerate(above_threshold_tco2):
    if val and start_index is None:
        start_index = i
    elif not val and start_index is not None:
        durations_above_threshold_tco2.append(time[i] - time[start_index])
        index_debut_above_threshold_tco2.append(start_index)
        start_index = None


# Calcul de la durée passée en dessous du seuil pour SpO2
durations_below_threshold_spo2 = []
index_debut_below_threshold_spo2 = []
start_index = None
for i, val in enumerate(below_threshold_spo2):
    if val and start_index is None:
        start_index = i
    elif not val and start_index is not None:
        durations_below_threshold_spo2.append(time[i] - time[start_index])
        index_debut_below_threshold_spo2.append(start_index)
        start_index = None



# Calcul de la durée totale au-dessus du seuil de Tco2
total_duration_above_tco2_threshold = np.sum(smooth_Tcco2 > threshold_high_tcO2) / srate
# Calcul de la durée totale en dessous du seuil de SpO2
total_duration_below_spo2_threshold = np.sum(clean_Spo2 < threshold_low_SpO2) / srate



total_duration_seconds = time.size / srate
duree_enregistrement = convert_seconds_to_hh_mm_ss (total_duration_seconds)

print("Duree de l'enregistrement :",duree_enregistrement)

print("Nombre de passages de seuil pour tCO2:", num_threshold_crossings_tco2)
print("Durées passées au-dessus du seuil pour tCO2:", durations_above_threshold_tco2)
print("Indices des passages au-dessus du seuil pour tCO2 : ",index_debut_above_threshold_tco2)
print("Durée totale passée au-dessus du seuil de TCO2:",total_duration_above_tco2_threshold)


print("Nombre de passages de seuil pour SpO2:", num_threshold_crossings_spo2)
print("Durées passées en dessous du seuil pour SpO2:", durations_below_threshold_spo2)
print("Indices des passages en dessous du seuil pour SpO2 : ",index_debut_below_threshold_spo2)
print("Durée totale passée en dessous du seuil de SpO2:",total_duration_below_spo2_threshold)

fig, axs = plt.subplots(nrows = 2, sharex = True)

ax = axs[0]
ax.plot(time,raw_Spo2)
ax.plot(time, clean_Spo2)
ax.set_title(f'nombre de passages en dessous du seuil de SPO2 : {num_threshold_crossings_spo2} ')
ax.axhline(threshold_low_SpO2, color = 'r')

ax = axs [1]
ax.plot(time,raw_tco2)
ax.plot(time,smooth_Tcco2)
ax.set_title('TCO2')
ax.set_title(f'nombre de passages au dessus du seuil de TCO2 : {num_threshold_crossings_tco2} ')
ax.axhline(threshold_high_tcO2, color = 'r')

plt.show()
