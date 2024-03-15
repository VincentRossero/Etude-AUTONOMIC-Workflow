import physio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from tools import *


raw_tco2, srate = physio.read_one_channel('EEG_28.TRC',format='micromed', channel_name='tCO2+')
raw_Pnaz, srate = physio.read_one_channel('EEG_28.TRC',format='micromed', channel_name='Pnas+')
raw_Spo2, srate1 = physio.read_one_channel('EEG_28.TRC',format='micromed', channel_name='SpO2+')
raw_Sao2, srate = physio.read_one_channel('EEG_28.TRC',format='micromed', channel_name='SAO2+')




raw_Sao2 = abs(raw_Sao2)
smooth_Spo2 = replace_zeros_with_mean(raw_Spo2)




time = np.arange(0, raw_Pnaz.size/srate, 1 /srate) 

fig, axs = plt.subplots(nrows = 2, sharex = True)
fig.suptitle('TEST')

ax = axs[0]
ax.plot(time, raw_Pnaz)
ax.set_title('SAO2')

ax = axs[1]

ax.plot(time, smooth_Spo2)
#ax.plot(time,smooth_Spo)
ax.set_title('spo2')
ax.set_ylim(80,100)

plt.show ()

