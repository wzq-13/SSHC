import numpy as np
import time
import os 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import tqdm
import globalvar
from utils.utils import check_polygon_intersection, get_rect_points_vectorized, path_smoothness
from utils.prob import _create_objective_function, obj_fn, xy2xy_heading
from models.utils import create_model
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")
def path_clean(path, target):
    """
    清理路径，去除重复点和距离过近的点
    """
    # # 找到离终点最近的点
    dists_to_target = np.linalg.norm(path[:, :2] - target, axis=1)
    min_index = np.argmin(dists_to_target)
    path = path[:min_index + 1]
    # # print(path)

    i = 1
    while i < len(path):
        min_index = i-1
        min_dist = np.linalg.norm(path[i, :2] - path[min_index, :2])
        for j in range(0, i-1):
            dist = np.linalg.norm(path[j, :2] - path[i, :2])
            if dist < min_dist:
                min_dist = dist
                min_index = j
        if min_index != i-1:
            # 删掉从 min_index+1 到 i-1 的点
            path = np.delete(path, np.s_[min_index+1:i], axis=0)
            i = min_index + 1
        else:
            i += 1
    # return path
    cleaned_path = [path[0]]
    i=1
    while i < len(path)-1:
        next_p = None
        next_index = i
        for j in range(i, len(path)-1):
            dist = np.linalg.norm(path[j, :2] - cleaned_path[-1][:2])
            if dist >= 0.8:  # 保留距离大于阈值的点
                next_p = path[j]
                next_index = j
                break
        if next_p is not None:
            cleaned_path.append(next_p)
        i = next_index if next_index > i else i + 1  # 防止死循环
    cleaned_path.append(path[-1])  # 确保终点被添加
    return np.array(cleaned_path)
class PINN_Trainer:
    def __init__(self, config, train_loader, val_loader, test_loader=None, save_dir=None, load_dir=None, log_dir=None):
        """Initializes the Trainer with data, method, and configuration."""
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.violation_func = _create_objective_function()
        if load_dir is not None:
            checkpoint = torch.load(load_dir, map_location=DEVICE)
            load_config = checkpoint.get('config', None)
            self.config['hidden_dim'] = load_config.get('hidden_dim', self.config['hidden_dim'])
            self.config['dropout'] = load_config.get('dropout', self.config['dropout'])
            self.config['output_dim'] = load_config.get('output_dim', self.config['output_dim'])
        self.model = create_model(self.config, device=DEVICE)
        learning_rate = self.config['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.config['weight_decay'])
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['lr_milestones'], gamma=self.config['lr_gamma'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_decay_step'], gamma=self.config['lr_decay'])
        
        if self.save_dir is not None:
            print(f'Creating save directory at {self.save_dir}')
            os.makedirs(self.save_dir, exist_ok=True)
        if load_dir is not None:
            checkpoint = torch.load(load_dir, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'Model loaded from {load_dir}')
            
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
            
    def compute_loss(self, X_batch: torch.Tensor, Y_pred: torch.Tensor, epoch: int) -> Dict[str, torch.Tensor]:
        """Computes the loss for a batch."""
        obj_loss = obj_fn(X_batch, Y_pred, config=self.config)
        constraint_residuals = self.violation_func(X_batch, Y_pred)
        constraint_residuals = torch.mean(torch.abs(constraint_residuals))
        total_loss = obj_loss * self.config['obj_weight'] + constraint_residuals * self.config['constraint_weight'] if epoch >= self.config['constraint_start_epoch'] else obj_loss * self.config['obj_weight']
        return {
            'total_loss': total_loss,
            'obj_loss': obj_loss,
            'constraint_residuals': constraint_residuals
        }
        
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Trains the model for one epoch."""
        epoch_metrics = {'total_loss': 0.0, 'obj_loss': 0.0, 'constraint_residuals': 0.0}
        self.model.train()
        bar = tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.config['num_epochs']}")
        for X_batch in bar:
            for key in X_batch:
                X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
            
            self.optimizer.zero_grad()
            Y_pred = self.model(X_batch)
            loss_metrics = self.compute_loss(X_batch, Y_pred, epoch)
            loss = loss_metrics['total_loss']
            loss.backward()
            
            # Clamp gradients to avoid explosion
            if self.config['gradient_clipping']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['gradient_clipping_max_norm'])

            self.optimizer.step()
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                obj=f"{loss_metrics['obj_loss'].item():.4f}",
                cons=f"{loss_metrics['constraint_residuals'].item():.4f}"
            )
            for key in epoch_metrics:
                epoch_metrics[key] += loss_metrics[key].item()
        self.scheduler.step()
        
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def train(self, begin_epoch: int = 0):
        """Main training loop."""
        num_epochs = self.config['num_epochs']
        for epoch in range(begin_epoch, num_epochs):
            train_metrics = self.train_epoch(self.train_loader, epoch)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_metrics["total_loss"]:.4f}, obj Loss: {train_metrics["obj_loss"]:.4f}, Constraint Residual: {train_metrics["constraint_residuals"]:.4f}')
            if self.writer is not None:
                self.writer.add_scalar('Train/Total_Loss', train_metrics['total_loss'], epoch)
                self.writer.add_scalar('Train/Obj_Loss', train_metrics['obj_loss'], epoch)
                self.writer.add_scalar('Train/Constraint_Residuals', train_metrics['constraint_residuals'], epoch)
            if (epoch + 1) % self.config['eval_step'] == 0:
                val_metrics = self.evaluate(self.val_loader)
                print(f'--- Validation Loss: {val_metrics["total_loss"]:.4f}, obj Loss: {val_metrics["obj_loss"]:.4f}, Constraint Residual: {val_metrics["constraint_residuals"]:.4f}')
                if self.writer is not None:
                    self.writer.add_scalar('Val/Total_Loss', val_metrics['total_loss'], epoch)
                    self.writer.add_scalar('Val/Obj_Loss', val_metrics['obj_loss'], epoch)
                    self.writer.add_scalar('Val/Constraint_Residuals', val_metrics['constraint_residuals'], epoch)

            if (epoch + 1) % self.config['save_step'] == 0:
                self._save_model(epoch=epoch)

    def compute_score(self, X_batch: torch.Tensor, Y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes score."""
        Y_pred = Y_pred.view(Y_pred.size(0), -1, 7)  # (B, N, 7)
        Y_pred = Y_pred[:, :, :2]  # (B, N, 2)
        Y_cleaned = path_clean(Y_pred[0].detach().cpu().numpy(), X_batch['target'][0, :2].detach().cpu().numpy())  # (N_cleaned, 2)
        lengths = np.sum(np.linalg.norm(np.diff(Y_cleaned, axis=0), axis=1))

        xy_heading = xy2xy_heading(Y_pred)[0,1:,:]  # (N, 3)
        rect_points = get_rect_points_vectorized(xy_heading, width=globalvar.vehicle_geometrics_.vehicle_width, length=globalvar.vehicle_geometrics_.vehicle_length)  # (N, 4, 2)
        obstacles = X_batch['obstacles_vertices'][0]  # (M, 4, 2)
        collision = False
        for i in range(rect_points.shape[0]):
            for j in range(obstacles.shape[0]):
                if check_polygon_intersection(rect_points[i].cpu().numpy(), obstacles[j].cpu().numpy()):
                    collision = True
                    break
            if collision:
                break
        collision = 1.0 if collision else 0.0
        smoothness, curvature_score = path_smoothness(Y_pred.cpu().numpy()[0][:30])
        
        target = X_batch['target'][0, :2]
        distances = torch.norm(Y_pred[0] - target, dim=1)
        min_distance = distances.min().item()
        
        return {
            'length': lengths.item(),
            'collision': collision,
            'smoothness': smoothness,
            'curvature': curvature_score,
            'min_distance': min_distance
        }
        
        

    def test(self, data_loader: DataLoader) -> Dict[str, float]:
        test_metrics = {'average_time': 0.0, 'collision_rate':0.0, 'average_length':0.0, 'smoothness':0.0, 'curvature':0.0, 'min_distance':0.0, 'success': 0.0}
        self.model.eval()
        total_samples = 0
        
        # warm up
        with torch.no_grad():
            warm_num = 10
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_pred = self.model(X_batch)
                warm_num -= 1
                if warm_num <=0:
                    break
        
        with torch.no_grad():
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                # torch.cuda.synchronize()
                start_time = time.time()
                Y_pred = self.model(X_batch)
                # torch.cuda.synchronize()
                end_time = time.time()
                test_metrics['average_time'] += (end_time - start_time)
                score_metrics = self.compute_score(X_batch, Y_pred)
                test_metrics['average_length'] += score_metrics['length']
                test_metrics['collision_rate'] += score_metrics['collision']
                test_metrics['smoothness'] += score_metrics['smoothness']
                test_metrics['curvature'] += score_metrics['curvature']
                test_metrics['min_distance'] += score_metrics['min_distance']
                is_success = False
                if score_metrics['collision'] == 0.0 and score_metrics['min_distance'] < 2.0:
                    test_metrics['success'] += 1.0
                    is_success = True
                total_samples += 1
                print(f'Test Sample {total_samples}, Time: {(end_time - start_time)} seconds, Length: {score_metrics["length"]:.4f}, Collision: {score_metrics["collision"]}, Smoothness: {score_metrics["smoothness"]:.4f}, Curvature: {score_metrics["curvature"]:.4f}, Min Distance: {score_metrics["min_distance"]:.4f}, Success: {is_success}')

        for key in test_metrics:
            test_metrics[key] /= total_samples
        print("=== Test Results ===")
        print(f"Test Average Time per Batch: {test_metrics['average_time']:.4f} seconds")
        print(f"Test Average Length: {test_metrics['average_length']:.4f}")
        print(f"Test Collision Rate: {test_metrics['collision_rate']:.4f}")
        print(f"Test Smoothness: {test_metrics['smoothness']:.4f}")
        print(f"Test Curvature: {test_metrics['curvature']:.4f}")
        print(f"Test Minimum Distance to Obstacles: {test_metrics['min_distance']:.4f}")
        print(f"Test Success Rate: {test_metrics['success']:.4f}")

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a validation or test set."""
        eval_metrics = {'total_loss': 0.0, 'obj_loss': 0.0, 'constraint_residuals': 0.0}
        self.model.eval()
        with torch.no_grad():
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_pred = self.model(X_batch)
                loss_metrics = self.compute_loss(X_batch, Y_pred, epoch=self.config['constraint_start_epoch'] + 1)
                for key in eval_metrics:
                    eval_metrics[key] += loss_metrics[key].item()
        num_batches = len(data_loader)
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
            
        return eval_metrics
    
    def save_path_data(self, data_loader: DataLoader, path_data_dir='/mnt/sim/carla/carla-ue4-0.9.16/PythonAPI/examples/path_data/Soft') -> Dict[str, float]:
        self.model.eval()
        if not os.path.exists(path_data_dir):
            os.makedirs(path_data_dir)
        
        # 3. 正式测试循环
        with torch.no_grad():
            for batch_idx, X_batch in enumerate(data_loader):
                # 数据搬运
                save_path = os.path.join(path_data_dir, f'batch_{batch_idx}.npy')
                if os.path.exists(save_path):
                    continue
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                
                # 模型推理
                Y_final = self.model(X_batch)
                Y_final = Y_final.view(Y_final.size(0), -1, 7)  # (B, N, 7)
                Y_final = Y_final[:, :, :2]  # (B, N, 2)
                Y_final_numpy = Y_final[0].cpu().numpy()
                # 保存Y_final_numpy
                np.save(save_path, Y_final_numpy)
                print(f"Saved Y_final_numpy for batch {batch_idx}.")
    
    def _save_model(self, epoch: int):
        """Saves the model checkpoint."""
        if self.save_dir is not None:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, f'{self.save_dir}/epoch_{epoch}.pth')
            print(f'Model checkpoint saved at epoch {epoch} to {self.save_dir}')