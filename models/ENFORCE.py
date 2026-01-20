import numpy as np
import time
import os 
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import csv
import pandas as pd
from typing import Tuple, Callable, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.func import vmap, jacrev
import globalvar
from utils.utils import check_polygon_intersection, get_rect_points_vectorized, path_smoothness
from utils.prob import _create_objective_function, obj_fn, xy2xy_heading
from models.utils import create_model
RESULT_DIR = './test_hard2_logs'
CSV_FILE_PATH = os.path.join(RESULT_DIR, 'batch_details.csv')
SUMMARY_FILE_PATH = os.path.join(RESULT_DIR, 'final_summary.txt')
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def path_clean(path, target):
    '''
    remove points in path that are too close to each other
    '''
    dists_to_target = np.linalg.norm(path[:, :2] - target, axis=1)
    min_index = np.argmin(dists_to_target)
    path = path[:min_index + 1]

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
            path = np.delete(path, np.s_[min_index+1:i], axis=0)
            i = min_index + 1
        else:
            i += 1
    cleaned_path = [path[0]]
    i=1
    while i < len(path)-1:
        next_p = None
        next_index = i
        for j in range(i, len(path)-1):
            dist = np.linalg.norm(path[j, :2] - cleaned_path[-1][:2])
            if dist >= 0.8:
                next_p = path[j]
                next_index = j
                break
        if next_p is not None:
            cleaned_path.append(next_p)
        i = next_index if next_index > i else i + 1
    cleaned_path.append(path[-1])
    return np.array(cleaned_path)

class NeuralProjection(nn.Module):
    def __init__(self):
        super().__init__()
        
    def compute_batch_jacobian(self, data, y, constraints_fn):
        
        def single_constraint_fn(y_single, data_single):
            y_fake_batch = y_single.unsqueeze(0)
            data_fake_batch = {
                k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                for k, v in data_single.items()
            }
            constraints_output = constraints_fn(data_fake_batch, y_fake_batch)
            return constraints_output.squeeze(0)

        data_in_dims = {k: 0 for k in data.keys()}
        batch_jac_fn = vmap(jacrev(single_constraint_fn, argnums=0), in_dims=(0, data_in_dims))
        B = batch_jac_fn(y, data)
        return B
    @torch.compile
    def forward(self, data: Dict, y_pred: torch.Tensor, 
                constraints_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        constraints = constraints_fn(data, y_pred) # value: h(y)
        B = self.compute_batch_jacobian(data, y_pred, constraints_fn) 
        BB_T = torch.bmm(B, B.transpose(1, 2)) # (Batch, 10, 10)
        jitter = 1e-6 * torch.eye(BB_T.shape[1], device=BB_T.device).unsqueeze(0)
        L = torch.linalg.cholesky(BB_T + jitter)
        target = constraints.unsqueeze(-1) 
        z = torch.cholesky_solve(target, L) 
        delta = torch.bmm(B.transpose(1, 2), z).squeeze(-1)
        y_proj = y_pred - delta
        return y_proj
    

class AdaNP(nn.Module):
    def __init__(self, max_depth: int = 100, tol: float = 1e-6):
        super(AdaNP, self).__init__()
        self.max_depth = max_depth
        self.tol = tol
        self.projection_layer = NeuralProjection()
    
    def forward(self, data: Dict, y_pred: torch.Tensor, 
                constraints_fn: Callable) -> Tuple[torch.Tensor, int]:
        y_current = y_pred
        
        # --- 初始化最佳记录 ---
        # 初始状态设为 best
        best_y = y_current 
        min_residual = float('inf')
        best_depth = 0
        
        for i in range(self.max_depth):
            with torch.no_grad():
                constraints_val = constraints_fn(data, y_current)
                constraint_residual = torch.max(constraints_val)
            
            if constraint_residual < self.tol:
                return y_current, i
            
            if constraint_residual < min_residual:
                min_residual = constraint_residual
                best_y = y_current
                best_depth = i
            
            y_proj = self.projection_layer(data, y_current, constraints_fn)
            y_current = y_proj

        return best_y, best_depth

class ENFORCE(nn.Module):
    def __init__(self, backbone: nn.Module,
                 max_depth: int = 100, inference_tol: float = 1e-6,
                 training_tol: float = 1e-4):
        super(ENFORCE, self).__init__()
        
        self.backbone = backbone
        self.adanp = AdaNP(max_depth=max_depth, tol=inference_tol)
        self.training_tol = training_tol
        self.projection_layer = NeuralProjection()
        
        # 训练状态跟踪
        self.adaptive_training = True
    def forward(self, data: torch.Tensor, constraints_fn: Callable) -> Tuple[torch.Tensor, dict]:
        # Backbone Network Prediction
        y_pred = self.backbone(data)
        info = {
            'projection_depth': 0,
            'constraint_residual': 0.0,
            'projection_displacement': 0.0
        }
        
        y_final, projection_depth = self.adanp(data, y_pred, constraints_fn)
        info['projection_depth'] = projection_depth

        return y_final, info, y_pred
 
class ENFORCE_Trainer:
    def __init__(self, config, train_loader, val_loader, test_loader=None, save_dir=None, load_dir=None, log_dir=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.constraint_func = _create_objective_function()
        if load_dir is not None:
            checkpoint = torch.load(load_dir, map_location=DEVICE)
            load_config = checkpoint.get('config', None)
            self.config['hidden_dim'] = load_config.get('hidden_dim', self.config['hidden_dim'])
            self.config['dropout'] = load_config.get('dropout', self.config['dropout'])
            self.config['output_dim'] = load_config.get('output_dim', self.config['output_dim'])
            
        backbone = create_model(self.config, device=DEVICE)
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        if load_dir is not None:
            checkpoint = torch.load(load_dir, map_location=DEVICE)
            # 删掉backbone前缀
            model_dict = checkpoint['model_state_dict']
            backbone_dict = {}
            for k, v in model_dict.items():
                if k.startswith('backbone.'):
                    backbone_dict[k[len('backbone.'):]] = v
                else:
                    backbone_dict[k] = v

            backbone.load_state_dict(backbone_dict)
            load_config = checkpoint.get('config', None)
            self.config['hidden_dim'] = load_config.get('hidden_dim', self.config['hidden_dim'])
            print(f'Model loaded from {load_dir}')
        self.model = ENFORCE(
            backbone=backbone,
            max_depth=self.config['max_depth'],
            inference_tol=self.config['inference_tol'],
            training_tol=self.config['training_tol']
        ).to(DEVICE)
        learning_rate = self.config['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.config['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_decay_step'], gamma=self.config['lr_decay'])
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
    def compute_loss(self, X_batch: torch.Tensor, Y_pred: torch.Tensor, Y_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the loss for a batch."""
        obj_loss_pred = obj_fn(X_batch, Y_pred, config=self.config)
        proj_loss = torch.mean((Y_final.detach() - Y_pred)**2)
        constraint_residuals = self.constraint_func(X_batch, Y_final)
        constraint_residuals = torch.mean(torch.abs(constraint_residuals))
        total_loss = obj_loss_pred * self.config['obj_weight'] + self.config['proj_loss_weight'] * proj_loss + constraint_residuals * self.config.get('constraint_weight', 1.0)
        return {
            'total_loss': total_loss,
            'obj_loss': obj_loss_pred,
            'proj_loss': proj_loss,
            'constraint_residuals': constraint_residuals
        }
        

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Trains the model for one epoch."""
        epoch_metrics = {'total_loss': 0.0, 'obj_loss': 0.0, 'proj_loss': 0.0, 'constraint_residuals': 0.0}
        self.model.train()
        bar = tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.config['num_epochs']}")
        for X_batch in bar:
            for key in X_batch:
                X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
            self.optimizer.zero_grad()
            Y_final, info, Y_pred = self.model(X_batch, constraints_fn=self.constraint_func)
            loss_metrics = self.compute_loss(X_batch, Y_pred, Y_final)
            loss = loss_metrics['total_loss']
            loss.backward()
            # Clamp gradients to avoid explosion
            if self.config['gradient_clipping']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['gradient_clipping_max_norm'])
            self.optimizer.step()
            

            for key in epoch_metrics:
                if key in loss_metrics:
                    epoch_metrics[key] += loss_metrics[key].item()
            epoch_metrics.update({'projection_depth': epoch_metrics.get('projection_depth', 0.0) + info['projection_depth']})
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                obj=f"{loss_metrics['obj_loss'].item():.4f}",
                proj=f"{loss_metrics['proj_loss'].item():.6f}",
                cons=f"{loss_metrics['constraint_residuals'].item():.4f}",
                depth=f"{info['projection_depth']}"
            )
        self.scheduler.step()
        
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def train(self, begin_epoch: int = 0):
        """Main training loop."""
        num_epochs = self.config['num_epochs']
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(self.train_loader, epoch)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_metrics["total_loss"]:.4f}, obj Loss: {train_metrics["obj_loss"]:.4f}, Proj Loss: {train_metrics["proj_loss"]:.4f}, Constraint Residual: {train_metrics["constraint_residuals"]:.4f}, Avg Projection Depth: {train_metrics["projection_depth"]:.2f}')

            if (epoch + 1) % self.config['eval_step'] == 0:
                val_metrics = self.evaluate(self.val_loader)
                print(f'--- Validation Loss: {val_metrics["total_loss"]:.4f}, obj Loss: {val_metrics["obj_loss"]:.4f}, Proj Loss: {val_metrics["proj_loss"]:.4f}, Constraint Residual: {val_metrics["constraint_residuals"]:.4f}, Avg Projection Depth: {val_metrics["projection_depth"]:.2f}')
            if (epoch + 1) % self.config['save_step'] == 0:
                self._save_model(epoch=epoch)
        self._save_model(epoch=num_epochs)

    def compute_score(self, X_batch: torch.Tensor, Y_pred: torch.Tensor, Y_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes score."""
        Y_final = Y_final.view(Y_final.size(0), -1, 7)  # (B, N, 7)
        Y_final = Y_final[:, :, :2]  # (B, N, 2)
        Y_cleaned = path_clean(Y_final[0].detach().cpu().numpy(), X_batch['target'][0, :2].detach().cpu().numpy())  # (N_cleaned, 2)
        lengths = np.sum(np.linalg.norm(np.diff(Y_cleaned, axis=0), axis=1))

        xy_heading = xy2xy_heading(Y_final)[0,1:,:]  # (N, 3)
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
        
        smoothness, curvature_score = path_smoothness(Y_cleaned[:-1])
        
        target = X_batch['target'][0, :2]
        distances = torch.norm(Y_final[0] - target, dim=1)
        min_distance = distances.min().item()

        return {
            'length': lengths.item(),
            'collision': collision,
            'smoothness': smoothness,
            'curvature': curvature_score,
            'objective_distance': min_distance
        }
        
    def test(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_samples = 0
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
        
        headers = ['batch_id', 'time', 'length', 'collision', 'smoothness', 'curvature', 'objective_distance', 'projection_depth']
        with open(CSV_FILE_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        self.model.eval()
        total_samples = 0
        
        print("Starting Warm-up...")
        with torch.no_grad():
            warm_num = 20
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_final, info, Y_pred = self.model(X_batch, constraints_fn=self.constraint_func)
                warm_num -= 1
                if warm_num <= 0:
                    break
        print("Warm-up Done. Starting Test...")

        with torch.no_grad():
            for batch_idx, X_batch in enumerate(data_loader):
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                
                # torch.cuda.synchronize()
                start_time = time.time()
                Y_final, info, Y_pred = self.model(X_batch, constraints_fn=self.constraint_func)
                # torch.cuda.synchronize()
                end_time = time.time()
                time_cost = end_time - start_time
                
                score_metrics = self.compute_score(X_batch, Y_pred, Y_final)
                
                current_result = [
                    batch_idx,
                    time_cost,
                    score_metrics['length'].item() if torch.is_tensor(score_metrics['length']) else score_metrics['length'],
                    score_metrics['collision'].item() if torch.is_tensor(score_metrics['collision']) else score_metrics['collision'],
                    score_metrics['smoothness'].item() if torch.is_tensor(score_metrics['smoothness']) else score_metrics['smoothness'],
                    score_metrics['curvature'].item() if torch.is_tensor(score_metrics['curvature']) else score_metrics['curvature'],
                    score_metrics['objective_distance'].item() if torch.is_tensor(score_metrics['objective_distance']) else score_metrics['objective_distance'],
                    info['projection_depth'].item() if torch.is_tensor(info['projection_depth']) else info['projection_depth']
                ]
                
                with open(CSV_FILE_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(current_result)
                
                total_samples += 1
                
                print(f'Batch {batch_idx}, Time: {time_cost:.4f}s, Col: {current_result[3]}, Len: {current_result[2]:.4f}, Smooth: {current_result[4]:.4f}, Curv: {current_result[5]:.4f}, Obj Dist: {current_result[6]:.4f}')
                
                del Y_final, Y_pred, X_batch, score_metrics

        print(f"\nAll batches processed. Data saved to {CSV_FILE_PATH}")
        print("Calculating final statistics from file...")

        self.analyze_and_save_results()
    def analyze_and_save_results(self):
        """
        单独的函数：读取 CSV 并计算平均值
        """
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            success_df = df[df['collision'] == 0]
            avg_metrics = df.mean()
            avg_success_metrics = success_df.mean() if not success_df.empty else pd.Series(dtype=float)

            report = (
                "=== Test Results Summary ===\n"
                f"Total Batches: {len(df)}\n"
                f"Average Time:       {avg_metrics['time']:.4f} seconds\n"
                f"Average Length:     {avg_metrics['length']:.4f}\n"
                f"Collision Rate:     {avg_metrics['collision']:.4f}\n"
                f"Average Smoothness: {avg_metrics['smoothness']:.4f}\n"
                f"Average Curvature:  {avg_metrics['curvature']:.4f}\n"
                f"Avg Obj Distance:   {avg_metrics['objective_distance']:.4f}\n"
                f"Avg Projection Depth: {avg_metrics['projection_depth']:.2f}\n"
                "----------------------------\n"
                f"--- Successful Cases (No Collision) ---\n"
                f"Total Successful Batches: {len(success_df)}\n"
                f"Average Time:       {avg_success_metrics.get('time', 0.0):.4f} seconds\n"
                f"Average Length:     {avg_success_metrics.get('length', 0.0):.4f}\n"
                f"Average Smoothness: {avg_success_metrics.get('smoothness', 0.0):.4f}\n"
                f"Average Curvature:  {avg_success_metrics.get('curvature', 0.0):.4f}\n"
                f"Avg Obj Distance:   {avg_success_metrics.get('objective_distance', 0.0):.4f}\n"
                f"Avg Projection Depth: {avg_success_metrics.get('projection_depth', 0.0):.2f}\n"
                "============================\n"
            )
            
            print(report)
            
            with open(SUMMARY_FILE_PATH, 'w') as f:
                f.write(report)
                
        except Exception as e:
            print(f"Error calculating stats: {e}")
            print("Raw data is preserved in CSV file.")

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a validation or test set."""
        eval_metrics = {'total_loss': 0.0, 'obj_loss': 0.0, 'proj_loss': 0.0, 'constraint_residuals': 0.0}
        self.model.eval()
        for X_batch in data_loader:
            for key in X_batch:
                X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
            Y_final, info, Y_pred = self.model(X_batch, constraints_fn=self.constraint_func)
            loss_metrics = self.compute_loss(X_batch, Y_pred, Y_final)
            for key in eval_metrics:
                if key in loss_metrics:
                    eval_metrics[key] += loss_metrics[key].item()
            eval_metrics.update({'projection_depth': eval_metrics.get('projection_depth', 0.0) + info['projection_depth']})
        num_batches = len(data_loader)
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
            
        return eval_metrics
    
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