import os
import torch
import numpy as np

class UCIHARLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Inertial signals paths
        self.inertial_signals_dir = os.path.join(self.data_dir, split, 'Inertial Signals')
        
        # Signal file names
        self.signal_files = [
            'body_acc_x_', 'body_acc_y_', 'body_acc_z_',
            'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_',
            'total_acc_x_', 'total_acc_y_', 'total_acc_z_'
        ]
        
        # Load data
        self.X, self.y = self._load_data()

    def _load_data(self):
        X = []
        
        for signal_file in self.signal_files:
            file_path = os.path.join(self.inertial_signals_dir, f"{signal_file}{self.split}.txt")
            data = np.loadtxt(file_path)
            X.append(data)
        
        # Stack all the signals to form the input tensor
        X = np.stack(X, axis=-1)  # Shape: (samples, timesteps, signals)
        X = torch.tensor(X, dtype=torch.float32)
        
        # Load labels
        y_path = os.path.join(self.data_dir, self.split, f'y_{self.split}.txt')
        y = np.loadtxt(y_path, dtype=int) - 1  # Adjust labels to be 0-indexed
        y = torch.tensor(y, dtype=torch.long)
        
        return X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)

        return X, y

def get_dataloader(data_dir, batch_size=32, split='train', shuffle=True, transform=None):
    dataset = UCIHARLoader(data_dir=data_dir, split=split, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
