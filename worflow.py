


eeg_data = ['Cz', 'Fz']
eog_data = ['Eog1', 'Eog2']

channels = eeg_data + eog_data

data = []
for channel in channels:
    ch_data, srate = physio.read_one_channel('EEG_31.TRC', format='micromed', channel_name=channel)
    #data.append(ch_data)
    data.append(ch_data * 1e-6)

data_reref[eeg] = data 

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
