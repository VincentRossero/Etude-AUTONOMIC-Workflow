import physio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from tools import *
from matplotlib.ticker import FuncFormatter
from pandasgui import show

raw_abdo, srate = physio.read_one_channel('EEG_35.TRC',format='micromed', channel_name='abdo+')
raw_thor, srate =physio.read_one_channel('EEG_35.TRC',format='micromed', channel_name='thor+')
duree_debut= 21600
duree_max= 35000
frames_debut = 256 * duree_debut
frames_max = 256 * duree_max
raw_abdo = raw_abdo[frames_debut:frames_max]
raw_thor = raw_thor[frames_debut:frames_max]


time = np.arange(0, raw_abdo.size/srate, 1 /srate)
thor = iirfilt(raw_thor, srate, lowcut = 0.05, highcut = 2, show = False)
abdo = iirfilt(raw_abdo, srate, lowcut = 0.05, highcut = 2, show = False)



#je detecte au maximum un cycle par seconde, d'une hauteur minimum de (med+mad)

seconde = 1
med,mad = calcul_med_mad(thor)
height = med 

#pic recupéré sur signal à l'endroit

peak_inds, peaks_dict = scipy.signal.find_peaks(thor, height, distance=int(srate * seconde))
#peaks_heights = peaks_dict['peak_heights']
peaks_heights = thor[peak_inds]
inspi_array = np.column_stack((peak_inds/srate, peaks_heights))
inspi_expi = np.full((inspi_array.shape[0], 1), True, dtype=bool)
inspiration = np.hstack((inspi_array, inspi_expi))


#pic récupéré sur signal à l'envers

thor_neg = -thor
peak_inds_neg, peaks_dict_neg = scipy.signal.find_peaks(thor_neg,height,distance = int(srate * seconde))
peaks_heights_neg = thor[peak_inds_neg]
expi_array = np.column_stack((peak_inds_neg/srate, peaks_heights_neg))
inspi_expi = np.full((expi_array.shape[0], 1), False, dtype=bool)
expiration = np.hstack((expi_array,inspi_expi))


#on reconstruit la respiration 

respiration = np.vstack((inspiration, expiration))
respiration = respiration [respiration[:,0].argsort()]


#si presence de 2 pics expis ou inspi à l'affilé on supprime le moins "ample"

def ensure_alternance(respiration):
   
    to_remove = []

    # Parcourir le tableau en comparant chaque ligne avec la précédente
    prev_value = respiration[0, 2]  # Valeur booléenne de la première ligne
    prev_index = 0  # Indice de la première ligne
    for index in range(1, len(respiration)):
        current_value = respiration[index, 2]  # Valeur booléenne de la ligne actuelle

        # Si la valeur booléenne de la ligne actuelle est différente de la précédente
        if current_value != prev_value:
            # Si le groupe contient plus d'une ligne et que c'est une inspiration
            if prev_value:  
                max_index = np.argmax(respiration[prev_index:index, 1]) + prev_index
                to_remove.extend([i for i in range(prev_index, index) if i != max_index])
            else:  
                min_index = np.argmin(respiration[prev_index:index, 1]) + prev_index
                to_remove.extend([i for i in range(prev_index, index) if i != min_index])

            # Mettre à jour la valeur précédente et l'indice précédent
            prev_value = current_value
            prev_index = index

    # Supprimer les lignes à supprimer du tableau
    respiration = np.delete(respiration, to_remove, axis=0)

    return respiration



respiration_clean = ensure_alternance(respiration)



df = pd.DataFrame(respiration)
show(df)

df = pd.DataFrame(respiration_clean)
show(df)



#RESTE A CODER 
'''
cas des 4 inspis consecutives

etude de la variation d'amplitude pic à pic 

comment mieux définir les bornes de detections scipy 

certaines inspi son negatives et certaines expis sont positives 

'''
