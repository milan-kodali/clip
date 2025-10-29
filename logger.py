"""
training metrics logger
"""

import argparse
import csv
import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_dir='.cache/clip_data/logs', run_name='default'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.run_name = run_name
        self.log_file = os.path.join(log_dir, f'{run_name}.csv')
        self.metrics = []
        
        # init csv if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['step', 'train_loss', 'val_loss', 'lr', 'grad_norm', 'dt', 'imgs_per_sec'])
                writer.writeheader()
    
    def log(self, step, train_loss=None, val_loss=None, lr=None, grad_norm=None, dt=None, imgs_per_sec=None):
        metric = {
            'step': step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr,
            'grad_norm': grad_norm,
            'dt': dt,
            'imgs_per_sec': imgs_per_sec
        }
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'train_loss', 'val_loss', 'lr', 'grad_norm', 'dt', 'imgs_per_sec'])
            writer.writerow(metric)
        
        self.metrics.append(metric)
    
    def load_logs(self):
        metrics = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metric = {}
                    for key, value in row.items():
                        if value == '' or value == 'None':
                            metric[key] = None
                        elif key == 'step':
                            metric[key] = int(value)
                        else:
                            try:
                                metric[key] = float(value)
                            except ValueError:
                                metric[key] = value
                    metrics.append(metric)
        self.metrics = metrics
        return metrics
    
    def save_plot(self, path=None):
        if not self.metrics:
            self.load_logs()
        
        if not self.metrics:
            print("No metrics to plot")
            return
        
        # default save path: same as log file but with .png extension
        if path is None:
            path = self.log_file.replace('.csv', '.png')
        
        # extract data
        steps = [m['step'] for m in self.metrics]
        train_losses = [m['train_loss'] for m in self.metrics if m['train_loss'] is not None]
        train_steps = [m['step'] for m in self.metrics if m['train_loss'] is not None]
        val_losses = [m['val_loss'] for m in self.metrics if m['val_loss'] is not None]
        val_steps = [m['step'] for m in self.metrics if m['val_loss'] is not None]
        lrs = [m['lr'] for m in self.metrics if m['lr'] is not None]
        lr_steps = [m['step'] for m in self.metrics if m['lr'] is not None]
        
        # create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # plot loss
        if train_losses:
            axes[0].plot(train_steps, train_losses, label='train loss', alpha=0.7, linewidth=1.5)
        if val_losses:
            axes[0].plot(val_steps, val_losses, label='val loss', marker='o', markersize=4, linewidth=2)
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('loss')
        axes[0].set_title(f'loss curve ({self.run_name})')
        axes[0].set_ylim(bottom=0)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # plot lr
        if lrs:
            axes[1].plot(lr_steps, lrs, label='lr', color='green', linewidth=1.5)
            axes[1].set_xlabel('step')
            axes[1].set_ylabel('lr')
            axes[1].set_title('lr schedule')
            axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

def save_existing_log_plot(log_file, path=None):
    logger = Logger()
    logger.log_file = log_file
    logger.run_name = os.path.basename(log_file).replace('.csv', '')
    return logger.save_plot(path=path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training logs from a CSV file')
    parser.add_argument('log_file', type=str, help='Path to the log CSV file (e.g., .cache/clip_data/logs/run1.csv)')
    args = parser.parse_args()
    save_existing_log_plot(args.log_file)