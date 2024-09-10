from tsslbp import utils, config, model, datasets, loss

import torch
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

from time import time
def train(model, epochs, data_loader, optimizer, loss, device = "cuda"):
    model.train()
    model.to(device)
    
    total_acc = 0
    total_loss = 0
    total_len = 0
    
    with tqdm(data_loader, unit="batch") as nbatch:
        for data, targets in nbatch:
            data = data.to(torch.float32)
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            output_sspike = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            
            vloss = loss(outputs, targets)
            vloss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # model.weight_clipper()
            
            total_loss+= torch.sum(vloss).item()
            total_len+=len(targets)
            
            total_acc+=(
                (output_sspike.argmax(axis=1) == targets.cpu().numpy())
                .sum()
                .item()
            )
            
            nbatch.set_postfix_str(f"train acc: {total_acc / total_len:.3f}, train loss: {total_loss / total_len:.3f}")

def validation(model, data_loader, device):
    total_acc = 0
    total_len = 0
    
    model.eval()
    
    with tqdm(data_loader, unit="batch") as nbatch:
        for data, targets in nbatch:
            data = data.to(torch.float32)
            data, targets = data.to(device), targets.to(device)
            
            with torch.no_grad():
                outputs = model(data)
            
            output_sspike = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            total_len+=len(targets)
                
            total_acc+=(
                (output_sspike.argmax(axis=1) == targets.cpu().numpy())
                .sum()
                .item()
            )
            nbatch.set_postfix_str(f"val acc: {total_acc / total_len:.3f}")

def load_sample_cached(cached_dataloader):
    for i, (events, target) in enumerate(iter(cached_dataloader)):
        if i > 99:
            break
        
if __name__ == "__main__":
    utils.set_seed(7)
    
    # load dataset
    batch_size = 32
    train_loader, val_loader = datasets.get_gesture_dataset('data/', batch_size, n_steps = 5, dt = 1000)
    
    start_time = time()
    load_sample_cached(train_loader)
    print("elapsed", time() - start_time)
    # snn_model = model.DVSGestureNet(5, 1, n_steps = 5)
    
    # optimizer = torch.optim.AdamW(snn_model.parameters(), lr=0.0005)
    # spikeloss = loss.SpikeCountLoss(4, 1)
    
    # num_epochs = 100  # 원하는 에포크 수 설정

    # for epoch in range(num_epochs):
    #     train(snn_model, epoch, train_loader, optimizer, spikeloss, device = "cuda")
    #     validation(snn_model, val_loader, device = "cuda")