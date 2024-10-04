import os
import json

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from eva.utils import train, validation


def save_conf_matrix_fig(conf_matrix, save_path, map_label):
    plt.figure(figsize=(5, 5))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='none')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(np.arange(len(map_label)), map_label)
    plt.yticks(np.arange(len(map_label)), map_label)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()
    
def save_acc_metric_fig(precision_per_class, recall_per_class, f1_per_class, acc, save_path):
    precision_macro = precision_per_class.mean()
    recall_macro = recall_per_class.mean()
    f1_macro = f1_per_class.mean()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [acc, precision_macro, recall_macro, f1_macro]  # 임의의 예시 값


    plt.figure(figsize=(7, 5))
    plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    # 그래프에 제목과 라벨 추가
    plt.title('Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.ylim([0, 1])
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'acc_metric.png'))
    plt.close()
    
    
    
def cal_perf_metrics(gt, pred):
    num_gt_classes = len(np.unique(gt))
    
    conf_mat = np.zeros((num_gt_classes, num_gt_classes))
    for t, p in zip(gt, pred):
        conf_mat[t, p] +=1
        
    # Precision, Recall, F1 score 계산
    precision_per_class = np.zeros(num_gt_classes)
    recall_per_class = np.zeros(num_gt_classes)
    f1_per_class = np.zeros(num_gt_classes)

    for i in range(num_gt_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        precision_per_class[i] = TP / (TP + FP) if TP + FP > 0 else 0
        recall_per_class[i] = TP / (TP + FN) if TP + FN > 0 else 0
        f1_per_class[i] = 2 * (precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i]) if (precision_per_class[i] + recall_per_class[i]) > 0 else 0

    return conf_mat, precision_per_class, recall_per_class, f1_per_class
    
def create_report_dir(report_dir):
    """
    Recursively checks if the directory exists and increments the directory name if needed.
    Example: If 'model' exists, it will create 'model_0', 'model_1', etc.
    """
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        print(f"Created working directory: {report_dir}")
        return report_dir
    else:
        # If directory exists, check for a suffix like '_0', '_1', etc.
        if len(report_dir.split('_')) < 2:
            # If no suffix, start with '_0'
            report_dir = report_dir + '_0'
        else:
            # Increment the existing suffix
            base_name = '_'.join(report_dir.split('_')[:-1])
            count = int(report_dir.split('_')[-1]) + 1
            report_dir = f"{base_name}_{count}"
        
        # Recursively check for the new directory name
        return create_report_dir(report_dir)
    
class EVA:
    result_prefix = 'result'

    def __init__(self, model, optimizer, loss, dataloader, max_epochs, device):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataloader = dataloader
        self.max_epochs = max_epochs
        self.device = device
        self.report_dir = None
        self.learning_curves = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.summary = None
        
        self.metrics = {}
        
        self.report_dir = os.path.join(EVA.result_prefix, type(self.model).__name__)
        self.report_dir = create_report_dir(self.report_dir)

    def trainer(self):
        train_loader, val_loader = self.dataloader
        
        best_acc = 0
        self.summary = self.model.summary(train_loader.dataset[0][0].shape)
        self.summary.update({
            "batch_size": train_loader.batch_size,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "desired_count": getattr(self.loss, "desired_count", None),
            "undesired_count": getattr(self.loss, "undesired_count", None),
        })

        for epoch in range(self.max_epochs):
            train_acc, train_loss, train_len = train(self.model, epoch, train_loader, self.optimizer, self.loss, self.device)
            self.learning_curves['train_loss'].append(train_loss / train_len)
            self.learning_curves['train_acc'].append(train_acc / train_len)
            
            val_acc, val_len, (gt, preds) = validation(self.model, val_loader, self.device)
            self.learning_curves['val_acc'].append(val_acc / val_len)
            
            if (val_acc / val_len) > best_acc:
                best_acc = val_acc / val_len
                self.model.save_model(f'{self.report_dir}/best_model.pt')
                print(f"Best model saved @ epoch {epoch} (val_acc {(val_acc / val_len) * 100:.2f}%)")
                self.summary['best_val_acc'] = best_acc
                self.summary['best_epoch'] = epoch
                
                df = pd.DataFrame({'ground_truth': gt, 'predictions': preds})
                df.to_csv(os.path.join(self.report_dir, 'gt_preds.csv'))
                
                conf_mat, precision_macro, recall_macro, f1_macro = cal_perf_metrics(gt, preds)
                self.metrics['conf_matrix'] = conf_mat
                self.metrics['precision'] = precision_macro
                self.metrics['recall'] = recall_macro
                self.metrics['f1'] = f1_macro
                self.metrics['acc'] = best_acc
                self.metrics['map_label'] = train_loader.dataset.map_label
                
                print(conf_mat, precision_macro, recall_macro, f1_macro)
                
        self.generate_report()
        save_conf_matrix_fig(conf_mat, self.report_dir, list(train_loader.dataset.map_label.keys()))
        save_acc_metric_fig(precision_macro, recall_macro, f1_macro, best_acc, self.report_dir)

    def generate_report(self):
        # Save the summary dict as JSON
        summary_path = os.path.join(self.report_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=4)
        print(f"Summary saved to {summary_path}")

        # Plot and save learning curves
        self.plot_learning_curves()

        # Create markdown report
        self.generate_md_report(summary_path)

    def plot_learning_curves(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.learning_curves['train_loss'], label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')

        plt.subplot(1, 2, 2)
        plt.plot(self.learning_curves['train_acc'], label='Train Acc')
        plt.plot(self.learning_curves['val_acc'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        learning_curves_path = os.path.join(self.report_dir, "learning_curves.png")
        plt.savefig(learning_curves_path)
        plt.close()
        print(f"Learning curves saved to {learning_curves_path}")

    def generate_md_report(self, summary_path):
        md_report_path = os.path.join(self.report_dir, "report.md")
        
        # Load summary data for displaying as a table
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)

        # Create the Markdown file and write content
        with open(md_report_path, 'w') as f:
            f.write(f"# Training Report for {type(self.model).__name__}\n\n")
            f.write(f"## Model Summary\n")
            f.write("| Key | Value |\n")
            f.write("| --- | ----- |\n")
            
            # Write each key-value pair from the JSON summary as a table row
            for key, value in summary_data.items():
                f.write(f"| {key} | {value} |\n")
                
            f.write(f"\n## Best Model\n")
            f.write(f"[Download Best Model](best_model.pt)\n\n")
            
            f.write(f"\n## Learning Curves\n")
            f.write(f"![Learning Curves](learning_curves.png)\n\n")
            
            # Add Confusion Matrix and Accuracy Metric graphs
            f.write(f"## Confusion Matrix\n")
            f.write(f"![Confusion Matrix](confusion_matrix.png)\n\n")
            
            f.write(f"## Accuracy Metric\n")
            f.write(f"![Accuracy Metric](acc_metric.png)\n\n")

            # Add download link for gt_preds.csv
            f.write(f"## Ground Truth and Predictions\n")
            f.write(f"[Download Ground Truth & Predictions CSV](gt_preds.csv)\n\n")

            # Precision, Recall, F1-Score table
            f.write(f"## Precision, Recall, F1-Score (Per Class)\n")
            f.write("| No    | Class | Precision | Recall | F1-Score |\n")
            f.write("| ----- | ----- | --------- | ------ | -------- |\n")
            for idx, label_name in enumerate(self.metrics['map_label'].keys()):
                f.write(f"| {idx} |  {label_name} | {self.metrics['precision'][idx]:.4f} | {self.metrics['recall'][idx]:.4f} | {self.metrics['f1'][idx]:.4f} |\n")
            
            
        print(f"Markdown report with table saved to {md_report_path}")
