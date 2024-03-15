import physio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

#fonction permettant de transformer la pvalue en etoile 

def pval_stars(pval):
    if pval < 0.05 and pval >= 0.01:
        stars = '*'
    elif pval < 0.01 and pval >= 0.001:
        stars = '**'
    elif pval < 0.001 and pval >= 0.0001:
        stars = '***'
    elif pval < 0.0001:
        stars = '****'
    else:
        stars = 'ns'
    return stars


#fonction permettant de correler une variable par rapport à une autre avec spearman

def stats_quantitative(df, xlabel, ylabel,namex,namey, ax=None, corr_method = 'spearman'):
    if ax is None:
        fig, ax = plt.subplots()
 
    x = df[xlabel]
    y = df[ylabel]
 
    if corr_method == 'pearson':
        res_corr = scipy.stats.pearsonr(x, y)
        r = res_corr.statistic
    elif corr_method == 'spearman':
        res_corr = scipy.stats.spearmanr(x, y)
        r = res_corr.correlation
    pval_corr = res_corr.pvalue
    stars_corr = pval_stars(pval_corr)
 
    res_reg = scipy.stats.linregress(x, y)
    intercept = res_reg.intercept
    slope = res_reg.slope
    rsquare = res_reg.rvalue **2
    pval_reg = res_reg.pvalue
    stars_reg = pval_stars(pval_reg)
    ax.plot(x, intercept + slope*x, 'r', label=f'f(x) = {round(slope, 2)}x + {round(intercept, 2)}')
    ax.scatter(x = x, y=y, alpha = 0.8)
 
    ax.set_title(f'Correlation ({corr_method}) : {round(r, 3)}, p : {stars_corr}\nR² : {round(rsquare, 3)}, p : {stars_reg}')
    ax.set_xlabel(namex)
    ax.set_ylabel(namey)
    # ax.legend()
 
    return ax


#une fonction de lissage autour de la valeur moyenne d'un signal , en focntion d'une fenetre glissante de taille defini 
#signal , nombre d'ecghntillons pour la fenetre,  
def sliding_mean(sig, nwin, mode = 'same', axis = -1):
    """
    Sliding mean
    ------
    Inputs =
    - sig : nd array
    - nwin : N samples in the sliding window
    - mode : default = 'same' = size of the output (could be 'valid' or 'full', see doc scipy.signal.fftconvolve)
    - axis : axis on which sliding mean is computed (useful only in case of >= 1 dim)
    Output =
    - smoothed_sig : signal smoothed
    """
    if sig.ndim == 1:
        kernel = np.ones(nwin) / nwin
        smoothed_sig = scipy.signal.fftconvolve(sig, kernel , mode = mode)
        return smoothed_sig
    else:
        smoothed_sig = sig.copy()
        shape = list(sig.shape)
        shape[-1] = nwin
        kernel = np.ones(shape) / nwin
        smoothed_sig[:] = scipy.signal.fftconvolve(sig, kernel , mode = mode, axes = axis)
        return smoothed_sig



#seconde methode de lissage de signal à partir d'une bande passante 


def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter', verbose = False, show = False, axis = 0):

    """
    IIR-Filter of signal

    -------------------
    Inputs : 
    - sig : nd array
    - srate : sampling rate of the signal
    - lowcut : lowcut of the filter. Lowpass filter if lowcut is None and highcut is not None
    - highcut : highcut of the filter. Highpass filter if highcut is None and low is not None
    - order : N-th order of the filter (the more the order the more the slope of the filter)
    - ftype : Type of the IIR filter, could be butter or bessel
    - verbose : if True, will print information of type of filter and order (default is False)
    - show : if True, will show plot of frequency response of the filter (default is False)
    """

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    filtered_sig = scipy.signal.sosfiltfilt(sos, sig, axis=axis)

    if verbose:
        print(f'{ftype} iirfilter of {order}th-order')
        print(f'btype : {btype}')


    if show:
        w, h = scipy.signal.sosfreqz(sos,fs=srate, worN = 2**18)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.scatter(w, np.abs(h), color = 'k', alpha = 0.5)
        full_energy = w[np.abs(h) >= 0.99]
        ax.axvspan(xmin = full_energy[0], xmax = full_energy[-1], alpha = 0.1)
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig

#obtenir la fréquence respiratoire à partir de scipy, le signal sig doit etre pré traité
#height : hauteur à partir de laquelle on souhaite detecter pic 
#retourne un data frame , auquel  on peut ajouter des colones tels que TCO2 , a l'index ou on etait calcule les pics 



def frequenceScipy (sig,srate,height,seconde) :

    #indice en frame des pics detectés
    respi_peak_inds = scipy.signal.find_peaks(sig,height, distance = int(srate * seconde))[0]
    #indice en secondes des pics détectés
    resp_peak_times = respi_peak_inds / srate
    #calcul de la durée des cycles, on utilise gradient à la place de diff pour contourner une erreur de taille du tableau
    resp_cycle_duration = np.gradient(resp_peak_times)
    #calcul fréquence 
    resp_cycles_freq = 1/resp_cycle_duration

    data_scipy = pd.DataFrame ()
    data_scipy['inds']= respi_peak_inds
    data_scipy['times']= resp_peak_times
    data_scipy['duration']= resp_cycle_duration
    data_scipy['freq']= resp_cycles_freq
   
    return data_scipy

#calcul de la med et mad en fonction de mon signal 

def returnheight (sig) : 

    med = np.median(sig)
    mad = np.median(np.abs(sig - med)) / 0.6744897501960817
    height = med + 1 * mad
    return height


def replace_zeros_with_mean(signal):
    cleaned_signal = np.copy(signal)  # Pour éviter de modifier le signal original

    # Calculer la moyenne des valeurs non nulles dans le signal
    non_zero_values = signal[signal != 0]
    mean_value = np.mean(non_zero_values)

    # Remplacer les zéros par la moyenne calculée
    cleaned_signal[cleaned_signal == 0] = mean_value

    return cleaned_signal

def clean_Sao(signal):
    cleaned_signal = np.copy(signal)  # Pour éviter de modifier le signal original


    non_zero_values_above_75 = signal[(signal > 79) & (~np.isnan(signal))]
    mean_value = np.mean(non_zero_values_above_75)

 
    cleaned_signal[(cleaned_signal <= 79) | np.isnan(cleaned_signal)] = mean_value
    return cleaned_signal
