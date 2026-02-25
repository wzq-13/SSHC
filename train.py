import yaml
import torch
from models.ENFORCE import ENFORCE_Trainer
from models.SoftNet import PINN_Trainer
from DataLoader.dataload import My_Dataset
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random
import argparse
import os
def main():
    parser = argparse.ArgumentParser(description="ENFORCE Trainer/Tester")
    parser.add_argument('--mode', type=str, default='train', choices=['pre_train','train', 'test'],
                        help="pre_train or train or test")
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help="Directory to save the model checkpoints")
    parser.add_argument('--load_dir', type=str, default=None,
                        help="Directory to load the model checkpoints")
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help="Directory to save the training logs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    config_path = 'configs/default.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    batch_size = 512
    data_dir = './dataset'
    dataset = My_Dataset(data_dir=data_dir, length=200000)
    train_size = int(len(dataset) * 0.6)
    val_size = int((len(dataset) * 0.3))
    test_size = int((len(dataset) * 0.1))
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])
    test_dataset = torch.utils.data.Subset(dataset, range(0, test_size))
    val_dataset = torch.utils.data.Subset(dataset, range(test_size, test_size+val_size))
    train_dataset = torch.utils.data.Subset(dataset, range(test_size+val_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=16,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True,num_workers=16,persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,pin_memory=True,num_workers=16,persistent_workers=True)
    if args.mode == 'pre_train':
        print(f'Pre-training the model... Save directory: {args.save_dir} Log directory: {args.log_dir}')
        trainer = PINN_Trainer(config=config,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    save_dir=args.save_dir,
                    load_dir=args.load_dir,
                    log_dir=args.log_dir,
                )
        trainer.train()
    elif args.mode == 'train' or args.mode == 'test':
        trainer = ENFORCE_Trainer(config=config,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        save_dir=args.save_dir,
                        load_dir=args.load_dir,
                        log_dir=args.log_dir,
                    )
        if args.mode == 'train':
            print(f'Training the model... Save directory: {args.save_dir} Log directory: {args.log_dir}')
            trainer.train()
        elif args.mode == 'test':
            print(f'Testing the model... Load directory: {args.load_dir}')
            trainer.test()

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()