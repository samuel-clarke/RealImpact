import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from pydub.utils import db_to_float, ratio_to_db
import scipy.io.wavfile
from scipy.spatial.transform import Rotation
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, help='foo help')
args = parser.parse_args()

data_dir = args.directory
MAX_INT16 = 32767
NUM_MICROPHONES = 15
MIC_BAR_LENGTH = 1890 - 70

def normalize_gains(signal, gains):
    signal_fl = signal.astype(np.float32)
    for i in range(signal_fl.shape[0]):
        signal_fl[i, :] *= db_to_float(np.max(gains) - gains[i])
    return signal_fl

def window_hammer_signal(signal, threshold_ratio):
    # Here I am finding the max value in the hamnmer signal, then looking 
    # for where the signal is below a threshold on either side, i.e. when 
    # there is no force being recorded. Those samples are then set to 0 as 
    # we know they are noise. 
    max_val = np.max(signal, axis=1)
    max_val_idx = np.argmax(signal, axis=1)
    
    windowed_signal = np.zeros_like(signal)
    min_idx = np.zeros_like(max_val_idx)
    for i in range(signal.shape[0]):
        if (i % 100) == 0:
            print(i)
        curr_min_idx = max_val_idx[i] + np.argmax(signal[i, :(max_val_idx[i])+1:-1] < (threshold_ratio * max_val[i]))
        max_idx = max_val_idx[i] + np.argmax(signal[i, max_val_idx[i]:] < (threshold_ratio * max_val[i]))
        windowed_signal[i, 0:(max_idx-curr_min_idx)] = signal[i, curr_min_idx:max_idx]
        min_idx[i] = curr_min_idx
    
    return windowed_signal, min_idx

def deconvolve_hammer(signal, hammer):
    windowed_hammers, window_idx = window_hammer_signal(hammer, 0.01)
    windowed_hammers = np.repeat(windowed_hammers, NUM_MICROPHONES, axis=0).astype(np.float32)
    window_idx = np.repeat(window_idx, NUM_MICROPHONES, axis=0)
    deconvolved_signal = np.zeros_like(signal).astype(np.float32)
    deconvolved_signal = deconvolved_signal[:, np.min(window_idx):]
    for i in range(signal.shape[0]):
        if (i % 100) == 0:
            print(i)
        end_ind = (signal.shape[1]-window_idx[i])
        deconvolved_signal[i, :end_ind] = signal[i, window_idx[i]:]  
        deconvolved_signal[i, :end_ind] = np.real(np.fft.ifft(np.fft.fft(deconvolved_signal[i,:end_ind])
                                                       /np.fft.fft(windowed_hammers[i,:end_ind])))
    # Cut off end of deconvolved signal to remove ramp up at the end.
    deconvolved_signal = deconvolved_signal[:, :-int(fs*0.1)]
    return deconvolved_signal

def get_mic_world_space(angle, distance, ind):
    mic_z = -(MIC_BAR_LENGTH/2) + ind/14 * MIC_BAR_LENGTH
    mic_x = 230 + distance
    mic_y = -((45/2) + 20.95) * np.ones_like(angle)
    mic_points = np.vstack((mic_x, mic_y, mic_z)).transpose()
    rot = Rotation.from_euler('z', angle, degrees=True)
    pos_meters = rot.apply(mic_points) / 1000.0
    return pos_meters

vertex_dirs = glob.glob(os.path.join(data_dir, 'v_*'))
vertex_ids = []
hammers = []
hammer_gains = []
hammer_pads = []
audios = []
positions = []
hammer_gains = []
hammer_pads = []
mic_gains = []
mic_pads = []
for vd in vertex_dirs:
    print(vd)
    angle_offset = 0
    if 'degrees' in vd:
        result = re.search('.*_(.*)degrees', vd)
        print(result.group(1))
        angle_offset = int(result.group(1))
    df = pd.read_csv(os.path.join(vd, 'metadata.csv'))
    for i, row in df.iterrows():
        if row['valid'] == 1:
            angle = int(row['angle'])
            distance = int(row['distance'])
            rec_dir = os.path.join(vd, 'angle_%03i_distance_%04i'%(angle, distance))
            fs, data = scipy.io.wavfile.read(os.path.join(rec_dir, 'Force.wav'))
            hammers.append(data)
            vertex_ids.append(int(row['vertex_id']))
            hammer_gains.append(row['hammer_gain'])
            hammer_pads.append(row['hammer_pad'])
            for i in range(NUM_MICROPHONES):
                mic_gains.append(row['%i_gain'%(i+1)])
                mic_pads.append(row['%i_pad'%(i+1)])
                positions.append((angle+angle_offset, distance, i))
                fs, data = scipy.io.wavfile.read(os.path.join(rec_dir, 'Microphone_%i.wav'%(i+1)))
                audios.append(data)

audios = np.array(audios)
hammers = np.array(hammers)
positions = np.array(positions)
vertex_ids = np.array(vertex_ids)
mic_gains = np.array(mic_gains)
mic_pads = np.array(mic_pads, dtype=bool)
mic_adj_gains = mic_gains + 20 * np.logical_not(mic_pads)

audios_norm = normalize_gains(audios, mic_adj_gains)
hammers_norm = normalize_gains(hammers, hammer_gains)

deconvolved = deconvolve_hammer(audios_norm, hammers_norm)
deconvolved *= 0.99 / np.max(np.abs(deconvolved))

mesh = trimesh.load(os.path.join(data_dir, 'preprocessed/transformed.obj'), process=False, maintain_order=True)
vertex_positions = mesh.vertices[vertex_ids, :]
vertex_positions = np.repeat(vertex_positions, 15, axis=0)

listener_positions = get_mic_world_space(positions[:, 0], positions[:, 1], positions[:,2])

np.save(os.path.join(data_dir,'preprocessed/sounds.npy'), audios_norm / np.max(np.abs(audios_norm))/ 1.01)
np.save(os.path.join(data_dir,'preprocessed/hammers.npy'), hammers_norm)
np.save(os.path.join(data_dir,'preprocessed/deconvolved.npy'), deconvolved)
np.save(os.path.join(data_dir,'preprocessed/vertexID.npy'), np.repeat(vertex_ids, 15, axis=0))
np.save(os.path.join(data_dir,'preprocessed/listenerXYZ.npy'), listener_positions)
np.save(os.path.join(data_dir,'preprocessed/vertexXYZ.npy'), vertex_positions)
