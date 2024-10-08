import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import matplotlib.pyplot as plt

def set_seed(random_seed):
    """reproducible option

    Args:
        random_seed (int): seed value
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_uint8(data):
    """데이터를 0-255 사이로 정규화하고, uint8로 변환"""
    data_min = np.min(data)
    data_max = np.max(data)
    # 데이터가 이미 동일한 값일 경우 0으로 나눌 수 없으니, 이 경우 대비
    if data_max != data_min:
        normalized_data = (data - data_min) / (data_max - data_min) * 255.0
    else:
        normalized_data = np.zeros_like(data)
    return normalized_data.astype(np.uint8)

def plot_event_frame(event_data, file_name):
    n_step, ch, width, height = event_data.shape
    if ch == 1:
        # merge pol
        frame_concat = np.concatenate([event_data[j,0] for j in range(n_step)], axis=1)
    else:
        frame_concat = np.concatenate([
            np.stack([event_data[j, 0], event_data[j, 1], np.zeros((width, height))], axis=-1)  # R=polarity 0, G=polarity 1, B=0
            for j in range(n_step)
        ], axis=1)
    
    
    # 수평으로 이어 붙인 이미지들을 다시 수직으로 이어 붙임
    i8_data = to_uint8(frame_concat)
    
    plt.imsave(file_name, i8_data)
    
def train(model, epochs, data_loader, optimizer, loss, device = "cuda"):
    model.train()
    model.to(device)
    
    total_acc = 0
    total_loss = 0
    total_len = 0
    
    with tqdm(data_loader, unit="batch", ) as nbatch:
        nbatch.set_description(f'Epoch - {epochs}')
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