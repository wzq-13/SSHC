import yaml
import torch
import time
import argparse
from models.SoftNet import PINN_Trainer
from DataLoader.dataload import My_Dataset
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random
def main():
    config_path = 'configs/default.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    batch_size = 256
    data_dir = '/home/qian/dataset_V7/'
    dataset = My_Dataset(data_dir=data_dir, length=200000)
    train_size = int(len(dataset) * 0.6)
    val_size = int((len(dataset) * 0.3))
    test_size = int((len(dataset) * 0.1))
    test_dataset = torch.utils.data.Subset(dataset, range(0, test_size))
    val_dataset = torch.utils.data.Subset(dataset, range(test_size, test_size+val_size))
    train_dataset = torch.utils.data.Subset(dataset, range(test_size+val_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=16,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=16,persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=16,persistent_workers=True)

    time_now = time.strftime("%Y%m%d-%H%M%S")
    
    trainer = PINN_Trainer(config=config,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      save_dir=None,
                      load_dir=None,
                      log_dir=None, #f'logs/{time_now}'
                    )
    trainer.train()

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()