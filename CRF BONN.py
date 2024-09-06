import physio
import matplotlib.pyplot as plt
import numpy as np
import mne
import yasa
from tools import *
from matplotlib.ticker import FuncFormatter
from pandasgui import show
from compute_respiration_ceinture import *
#
channels = ['Cz', 'Fz']
data = []
for channel in channels:
    ch_data, srate = physio.read_one_channel('EEG_18.TRC', format='micromed', channel_name=channel)
    #data.append(ch_data)
    data.append(ch_data * 1e-6)


cz_data, fz_data = data
cz_fz_data = cz_data - fz_data
data.append(cz_fz_data)
ch_names = channels + ['Cz-Fz']

info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=['eeg']*len(channels) + ['eeg'])
raw = mne.io.RawArray(np.array(data), info)

raw.plot(duration=30, n_channels=len(channels), scalings='auto')
input("Appuyez sur Entrée pour quitter...")

raw.filter(1., 30., fir_design='firwin')
raw.set_eeg_reference('average')
eeg_name = 'Cz-Fz'


sl = yasa.SleepStaging(raw, eeg_name=eeg_name)
hypno = sl.predict()

raw_tco2, srate = physio.read_one_channel('EEG_18.TRC', format='micromed', channel_name='tCO2+')
raw_Spo2, srate = physio.read_one_channel('EEG_18.TRC', format='micromed', channel_name='SpO2+')
raw_resp, srate = physio.read_one_channel('EEG_18.TRC',format='micromed', channel_name='Pnaz+')
raw_abdo, srate = physio.read_one_channel('EEG_18.TRC',format='micromed', channel_name='abdo+')

raw_resp=-raw_resp
abdo = iirfilt(raw_abdo, srate, lowcut = 0.05, highcut = 2, show = False)

time = np.arange(0, raw_tco2.size / srate, 1 / srate) 
time_sec = np.arange(len(hypno)) * 30   #Multiplier par 30 pour obtenir le temps en secondes

parameters = physio.get_respiration_parameters('human_airflow') 
parameters['cycle_detection']['inspiration_adjust_on_derivative'] = True
resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameters=parameters)
inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values

peak_inds_a, peak_inds_neg_a = detection_respi (abdo,srate)
inspi_clean_a, expi_clean_a =interleave_insp_exp(peak_inds_neg_a,peak_inds_a)
respi_abdo = compute_respi_features (abdo,inspi_clean_a,expi_clean_a,srate)

#cycle de plus de 10 secondes
x=10
mask_apnea= resp_cycles['cycle_duration']>x
resp_cycles_a=resp_cycles[mask_apnea]
resp_cycles_a['total_duration']= resp_cycles_a['cycle_duration']
resp_cycles_a['apnee']=True

#cycle à faible amplitude
amplitude_moyenne = resp_cycles['total_amplitude'].mean()
seuil_apnee = amplitude_moyenne * 0.1
mask_amplitude=resp_cycles['total_amplitude']<seuil_apnee
resp_amplitude = resp_cycles[mask_amplitude]

index_amplitude = resp_amplitude.index
# Convertir les index en un DataFrame pour utiliser les opérations de décalage
index_df = pd.DataFrame(index_amplitude, columns=['index'])

# Créer une colonne qui indique les débuts de groupes consécutifs
index_df['group'] = (index_df['index'] != index_df['index'].shift() + 1).cumsum()

# Grouper par la colonne des groupes et récupérer les groupes d'indices consécutifs
consecutive_groups = index_df.groupby('group')['index'].apply(list)

result_indices = []
total_durations = []

# Parcourir les groupes et vérifier la somme des 'cycle_duration'

for group in consecutive_groups:
    
    total_duration = resp_amplitude.loc[group, 'cycle_duration'].sum()

    
    if total_duration > 10.0:
        # Ajouter le premier indice du groupe à la liste des résultats
        result_indices.append(group[0])
        total_durations.append(total_duration)

resp_apnee_ampl = resp_amplitude.loc[result_indices]
resp_apnee_ampl['total_duration'] = pd.Series(total_durations, index=result_indices)
resp_apnee_ampl['apnee']=False

inspi_index_to_drop = resp_apnee_ampl ['inspi_index'].tolist()
resp_cycles_a = resp_cycles_a[~resp_cycles_a['inspi_index'].isin(inspi_index_to_drop)]


#formation du tableau d'apnee final

total_apnea=pd.concat([resp_cycles_a,resp_apnee_ampl], axis = 0,ignore_index=True)
total_apnea = total_apnea.sort_values(by='inspi_index')

# ON ENELEVE LES PARTIES NON DESIREE 

mask = (total_apnea['inspi_time'] >= 4460) & (total_apnea['inspi_time'] <= 17000)
total_apnea = total_apnea[mask]
 

total_apnea = total_apnea.reset_index(drop=True)

# SI TROP D ARTEFACTS 

indices_to_keep = [1,2,3]
total_apnea = total_apnea.loc[total_apnea.index.isin(indices_to_keep)]

# SI BEAUCOUP D APNEE 

# indices_a_supprimer = [2,3]
# total_apnea = total_apnea.drop(indices_a_supprimer)


time_sec_hypno = np.arange(len(hypno)) * 30
def get_hypno_value(time_point):
    # Trouver l'indice le plus proche dans time_sec_hypno
    idx = np.searchsorted(time_sec_hypno, time_point, side='left')
    # Retourner la valeur de hypno correspondant
    return hypno[idx] if idx < len(hypno) else np.nan

# Ajouter une colonne avec les valeurs de l'hypnogramme à total_apnea
total_apnea['hypno_value'] = total_apnea['inspi_time'].apply(get_hypno_value)

show (total_apnea)





raw_tco2 = -raw_tco2
smooth_Tcco2 = iirfilt(raw_tco2,srate,highcut=1) 

#SPO2 et TCO2 en fonction de l'hypnogramme 

fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(10, 8))

axs[0].plot(time_sec, hypno, color='black')

axs[0].set_title('Hypnogramme')
axs[0].set_ylabel('Stades de sommeil')


#axs[1].plot(time, raw_tco2, color='blue')
axs[1].plot(time,smooth_Tcco2,color='blue')

axs[1].axhline(50, color='red')
axs[1].set_title('TCO2')
axs[1].set_ylabel('TCO2 (mmHh)')

axs[2].plot(time, raw_Spo2, color='green')

axs[2].axhline(90, color='red')
axs[2].set_ylim(85, 100)
axs[2].set_title('SPO2')
axs[2].set_ylabel('SPO2 (%)')

plt.tight_layout()
plt.show()

#visualisation des apnées

fig,axs = plt.subplots(nrows=3,sharex=True)

ax=axs[0]

ax.plot(time,resp)
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')

for index, row in total_apnea.iterrows():
    inspi_time = row['inspi_time']
    if row['apnee']:
        ax.axvline(inspi_time, color='k', lw=2, linestyle='--')
    else:
        ax.axvline(inspi_time, color='green', lw=2, linestyle='--')

ax=axs[1]
ax.plot (time,abdo)
ax.scatter(time[inspi_clean_a],abdo[inspi_clean_a],color='g')
ax.scatter(time[expi_clean_a],abdo[expi_clean_a],color='r')

ax=axs[2]
# ax.plot(time_sec, hypno, color='black')
ax.plot(time, raw_Spo2, color='green')
ax.axhline(90, color='red')
ax.set_ylim(85, 100)
ax.set_title('SPO2')
ax.set_ylabel('SPO2 (%)')

plt.show()

# Étape 1 : Définir la plage de temps d'intérêt
start_time = 4460
end_time = 17000

# Étape 2 : Créer un masque booléen pour sélectionner uniquement les temps dans la plage
mask = (time_sec_hypno >= start_time) & (time_sec_hypno <= end_time)

# Filtrer l'hypnogramme et les temps pour ne garder que cette plage
filtered_hypno = hypno[mask]
filtered_time_sec = time_sec_hypno[mask]

# Vérification : Combien de stades de sommeil sont dans cette plage ?
print(f"Nombre d'éléments filtrés dans l'hypnogramme : {len(filtered_hypno)}")

# Étape 3 : Calculer les durées pour chaque stade de sommeil
durations_filtered = {}

# Stades de sommeil et leurs correspondances de code
stages = ['W', 'N1', 'N2', 'N3', 'R']
time_in_each_stage = {}

for stage in stages:
    # Filtrer l'hypnogramme pour le stade donné
    stage_mask = filtered_hypno == stage
    time_in_stage = np.sum(stage_mask) * 30 / 60  # Multiplier par 30 pour convertir en secondes puis diviser par 60 pour les minutes
    
    time_in_each_stage[stage] = time_in_stage

# Afficher les résultats
for stage, time in time_in_each_stage.items():
    print(f"Temps passé dans le stade {stage} (entre 4460s et 17000s) : {time:.2f} minutes")
