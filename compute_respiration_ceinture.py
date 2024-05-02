'''

Attention 
Dans la fonction interleave_insp_exp
Tel que les fonctiosn ont ete concus , il faut rentrer les arguments dans cet ordre
Expi, Inspi 

'''
import physio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from tools import *
from matplotlib.ticker import FuncFormatter
from pandasgui import show


def detection_respi (abdo,srate,seconde=1) : 

    med,mad = calcul_med_mad(abdo)
    height = med + 0*mad 

    peak_inds = scipy.signal.find_peaks(abdo, height, distance=int(srate * seconde))[0]
    peak_inds_neg = scipy.signal.find_peaks(-abdo,height,distance = int(srate * seconde))[0]
    return peak_inds, peak_inds_neg

def _ensure_interleave(ind0, ind1, remove_first=True):
    """
    Clean ind0 so they are interlevaed into ind1.
    """
    keep_inds = np.searchsorted(ind1, ind0,  side='right')
    keep = np.ones(ind0.size, dtype=bool)
    ind_bad = np.flatnonzero(np.diff(keep_inds) == 0)
    if remove_first:
        keep[ind_bad] = False
    else:
        keep[ind_bad + 1] = False
    ind0_clean = ind0[keep]
    return ind0_clean
 
 
def interleave_insp_exp(ind_insp, ind_exp, remove_first_insp=True, remove_first_exp=False):
    """
    Ensure index of inspiration and expiration are interleaved.
 
    Ensure also that it start and stop with inspiration so that ind_insp.size == ind_exp.size + 1
    """
 
    ind_exp = _ensure_interleave(ind_exp, ind_insp, remove_first=remove_first_exp)
 
    ind_insp = _ensure_interleave(ind_insp, ind_exp, remove_first=remove_first_insp)
 
 
    if np.any(ind_exp < ind_insp[0]):
        ind_exp = ind_exp[ind_exp>ind_insp[0]]
 
    if np.any(ind_exp > ind_insp[-1]):
        ind_exp = ind_exp[ind_exp<ind_insp[-1]]
 
    # corner cases several ind_insp at the beginning/end
    n = np.sum(ind_insp < ind_exp[0])
    if n > 1:
        ind_insp = ind_insp[n - 1:]
    n = np.sum(ind_insp > ind_exp[-1])
    if n > 1:
        ind_insp = ind_insp[: - (n - 1)]
    return ind_insp, ind_exp

def compute_abdo_features (sig,inspi_ind_clean,expi_ind_clean,srate) :
    abdo_features = pd.DataFrame ()
    abdo_features['inspi_index'] = inspi_ind_clean[:-1]
    abdo_features['expi_index'] = expi_ind_clean
    abdo_features['next_inspi_index'] = inspi_ind_clean[1:]
    abdo_features['inspi_time'] = abdo_features['inspi_index']/srate
    abdo_features['expi_time'] = abdo_features['expi_index']/srate
    abdo_features['next_inspi_time'] = abdo_features['next_inspi_index']/srate
    abdo_features['inspi_duration'] = abdo_features ['expi_time'] - abdo_features['inspi_time']
    abdo_features['expi_duration'] = abdo_features ['next_inspi_time'] - abdo_features['expi_time']
    abdo_features['cycle_duration'] = abdo_features ['inspi_duration'] + abdo_features ['expi_duration']
    abdo_features['inspi_amplitude'] = sig[abdo_features['expi_index']]-sig[abdo_features['inspi_index']]
    abdo_features['expi_amplitude'] = sig[abdo_features['expi_index']]-sig[abdo_features['next_inspi_index']]
    abdo_features['cycle_amplitude'] = abdo_features['inspi_amplitude'] + abdo_features['expi_amplitude']


    return abdo_features


