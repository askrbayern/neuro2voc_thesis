import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AOIFixedShiftNeuralDataset(Dataset):
    def __init__(self, file_path: str, annotation_df: pd.DataFrame, sound_sample_rate: int, neural_sample_rate: int, pre: float, segment_length_samples: int, overlap_ratio: float, num_channels: int):
        logger.info("Loading data...")
        self.dat = np.memmap(file_path, dtype=np.int16, mode='r')
        self.dat = self.dat.reshape((num_channels, -1), order='F')
    

        self.pre = pre
        self.segment_length_samples = segment_length_samples
        self.shift = int(segment_length_samples * (1 - overlap_ratio))
        
        total_samples = self.dat.shape[1]
        
        logger.info("Generating segment indices...")
        self.segment_indices = []
        
        # for each onset, generate:
        for idx, row in annotation_df.iterrows():
            # the original time of annotation df is 20khz
            # we need to convert it to 30khz to match neuropixel sampling rate
            onset_time = row['onset'] * (neural_sample_rate / sound_sample_rate)
            duration_time = row['duration'] * (neural_sample_rate / sound_sample_rate)
            start_time = onset_time - self.pre  
            end_time = onset_time + duration_time 

            start_sample = int(start_time)
            end_sample = int(end_time)

            assert start_sample >= 0

            # moving window
            current_start = start_sample
            while current_start + segment_length_samples <= end_sample:
                self.segment_indices.append((current_start, current_start + segment_length_samples))
                current_start += self.shift

            if current_start < end_sample:
                self.segment_indices.append((current_start, min(current_start + segment_length_samples, total_samples)))
        
        logger.info(f"Total segments generated: {len(self.segment_indices)}")
        
    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):
        start, end = self.segment_indices[idx]
        segment = self.dat[:, start:end]
        return torch.from_numpy(segment.astype(np.float32))

def create_and_save_AOI_dataset(file_path, annotation_df, sound_sample_rate, neural_sample_rate, pre, segment_length, overlap_ratio, output_dir, num_channels):
    # segment length in milliseconds
    segment_length_samples = int(segment_length * neural_sample_rate / 1000)
    pre_samples = int(pre * neural_sample_rate / 1000)
    dataset = AOIFixedShiftNeuralDataset(file_path, annotation_df, sound_sample_rate, neural_sample_rate, pre_samples, segment_length_samples, overlap_ratio, num_channels)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving {len(dataset)} segments to {output_dir}...")
    for i in range(len(dataset)):
        segment = dataset[i]
        torch.save(segment, os.path.join(output_dir, f"segment_{i:06d}.pt"))
        if (i + 1) % 100 == 0:
            logger.info(f"Saved {i + 1} segments")
    
    logger.info("Dataset creation complete!")

if __name__ == "__main__":
    data_dir = "/mnt/m/neuro2voc/task-5"
    neural_data_file = 'neural_data_2_30k.bin'
    annotation_file = 'annotations_raw_g2_t0.nidq.bin.csv'

    sound_sample_rate = 20000  # 20kHz
    neural_sample_rate = 30000  # 30kHz
    pre = 60  # 60ms before event onset
    segment_length = 30  # segment length in ms
    num_channels = 75  # number of channels
    overlap_ratio = 1/3  # 1/3 overlap

    output_folder = f'AOI_30kHz_{pre}ms_pre_{segment_length}ms_segment_overlap_{overlap_ratio:.2f}'
    file_path = os.path.join(data_dir, neural_data_file)
    output_dir = os.path.join(data_dir, output_folder)	

    annotations_df = pd.read_csv(os.path.join(data_dir, annotation_file))
    filtered_df = annotations_df[annotations_df['cluster_id'].between(2, 8)]

    create_and_save_AOI_dataset(file_path, filtered_df, sound_sample_rate, neural_sample_rate, pre, segment_length, overlap_ratio, output_dir, num_channels)