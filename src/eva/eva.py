import os
import json
import matplotlib.pyplot as plt

from eva.utils import train, validation


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
            
            val_acc, val_len = validation(self.model, val_loader, self.device)
            self.learning_curves['val_acc'].append(val_acc / val_len)
            
            if (val_acc / val_len) > best_acc:
                best_acc = val_acc / val_len
                self.model.save_model(f'{self.report_dir}/best_model.pt')
                print(f"Best model saved @ epoch {epoch} (val_acc {(val_acc / val_len) * 100:.2f}%)")
                self.summary['best_val_acc'] = best_acc
                self.summary['best_epoch'] = epoch

        self.generate_report()

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
            f.write(f"### Summary Table\n\n")
            f.write("| Key | Value |\n")
            f.write("| --- | ----- |\n")
            
            # Write each key-value pair from the JSON summary as a table row
            for key, value in summary_data.items():
                f.write(f"| {key} | {value} |\n")
            
            f.write(f"\n## Learning Curves\n")
            f.write(f"![Learning Curves](learning_curves.png)\n\n")
        print(f"Markdown report with table saved to {md_report_path}")
