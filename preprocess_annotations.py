import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, help='foo help')
parser.add_argument('--num_microphones', type=int, default=15, help='Random seed')
args = parser.parse_args()

data_dir = args.directory
NUM_MICROPHONES = args.num_microphones

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
            vertex_ids.append(int(row['vertex_id']))
            for i in range(NUM_MICROPHONES):
                positions.append((angle+angle_offset, distance, i))

positions = np.array(positions)

print(positions[0, :])
print(positions.shape)
np.save(os.path.join(data_dir,'preprocessed/angle.npy'), positions[:,0])
np.save(os.path.join(data_dir,'preprocessed/distance.npy'), positions[:,1])
np.save(os.path.join(data_dir,'preprocessed/micID.npy'), positions[:,2])
