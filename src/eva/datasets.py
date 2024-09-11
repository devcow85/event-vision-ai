from torch.utils.data import DataLoader, Dataset

import tonic
import tonic.transforms as transforms

import os

from eva.prophesee_parser import PSEELoader

class NCARS(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform

        self.file_paths = {}
        
        self.map_label = {
            "cars" : 0,
            "background" : 1
        }
        
        if train:
            sub_path = 'n-cars_train'
        else:
            sub_path = 'n-cars_test'

        self.root_dir = os.path.join(root, sub_path)
        
        for path, _, files in os.walk(self.root_dir):
            files.sort()
            for filename in files:
                if filename.endswith("dat"):
                    full_path = os.path.join(path, filename)
                    self.file_paths[full_path] = path.split("/")[-1]
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path, label = list(self.file_paths.items())[idx]        
        # load event data using Prophsee foramtter
        event_obj = PSEELoader(img_path)
        data = event_obj.load_n_events(event_obj.event_count())
        
        if self.transform:
            data = self.transform(data)
        
        return data, self.map_label.get(label)