import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedShiftNeuralDataset(Dataset):
    def __init__(self, file_path: str, segment_length_samples: int, overlap_ratio: float, num_channels: int):
        logger.info("Loading data...")
        self.dat = np.memmap(file_path, dtype=np.int16, mode='r')
        self.dat = self.dat.reshape((num_channels, -1), order='F')

        self.segment_length_samples = segment_length_samples
        self.shift = int(segment_length_samples * (1 - overlap_ratio))
        
        total_samples = self.dat.shape[1]
        
        logger.info("Generating segment indices...")
        self.segment_indices = []
        
        current_start = 0
        while current_start + segment_length_samples <= total_samples:
            self.segment_indices.append((current_start, current_start + segment_length_samples))
            current_start += self.shift

        if current_start < total_samples:
            self.segment_indices.append((current_start, total_samples))
        
        logger.info(f"Total segments generated: {len(self.segment_indices)}")
        
    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):
        start, end = self.segment_indices[idx]
        segment = self.dat[:, start:end]
        return torch.from_numpy(segment.astype(np.float32))

def create_and_save_dataset(file_path, segment_length, overlap_ratio, output_dir, num_channels):
    dataset = FixedShiftNeuralDataset(file_path, segment_length, overlap_ratio, num_channels)
    
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

    segment_length = 500  # segment length in samples (300ms at 30kHz)
    num_channels = 75  # number of channels
    overlap_ratio = 1/3  # 1/3 overlap

    output_folder = f'notAOI_dataset_{segment_length}samples_segment_length_{overlap_ratio:.2f}_overlap'
    file_path = os.path.join(data_dir, neural_data_file)
    output_dir = os.path.join(data_dir, output_folder)	

    create_and_save_dataset(file_path, segment_length, overlap_ratio, output_dir, num_channels)