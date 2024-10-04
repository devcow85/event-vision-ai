import torch 

from torch.utils.data import DataLoader
import tonic.transforms as transforms

from eva import transforms as eva_transforms
from eva.datasets import PropheseeGen4CLSTEST

from eva import model
from eva.eva import cal_perf_metrics

if __name__ == "__main__":
    transform = transforms.Compose([
        eva_transforms.TemporalCrop(time_window = 99_000, padding_enable = True),
        # transforms.Denoise(filter_time=5500),
        eva_transforms.ToFrameAuto(n_time_bins = 5, aspect_ratio = False),
        eva_transforms.MergeFramePolarity(),
        eva_transforms.EventFrameResize((80,80)),
        eva_transforms.EventNormalize(mean=(128,), std=(1,))
    ])
    
    test_ds = PropheseeGen4CLSTEST(root = '/data/event-data/99.Prophesee_Gen4_testdata_100_BusRemake', transform=transform)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=40, num_workers=8)
    
    
    # load model
    snn_model = model.PPGen4NetMini128(5, 1, 5, "result/PPGen4NetMini128_3/best_model.pt")
    snn_model.eval()
    snn_model.cuda()
    
    total_acc = 0
    total_len = 0
    
    all_targets = []
    all_preds = []
    
    # # test loop
    for data, targets in test_loader:
        data = data.to(torch.float32)
        data, targets = data.cuda(), targets.cuda()
        
        with torch.no_grad():
            outputs = snn_model(data)
        
        output_spike = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
        
        preds = output_spike.argmax(axis=1)
        
        total_len+=len(targets)
        total_acc+=(
            (preds == targets.cpu().numpy())
            .sum()
            .item()
        )
        
        all_targets.extend(targets.detach().cpu().numpy())  # Ground truth values
        all_preds.extend(preds)  # Predicted values
    
    conf_mat, precision_per_class, recall_per_class, f1_per_class = cal_perf_metrics(all_targets, all_preds)
    print(f"Accuracy : {total_acc/total_len*100:.2f}%")
    print("Confusion Matrix")
    print(conf_mat)
    
    #################################################
    ## 0.test sample verification trace generation ##
    #################################################
    
    # import numpy as np
    # ids = [0,100,200,300,400]
    # input_chunk = np.array([test_ds[idx][0] for idx in ids])
    # target_chunk = np.array([test_ds[idx][1] for idx in ids])
    
    # input_tensor = torch.tensor(input_chunk).to(torch.float32).cuda()
    
    # snn_model.register_hook()
    
    # outputs = snn_model(input_tensor)
    
    # output_spike = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
    # preds = output_spike.argmax(axis=1)
    
    # print(preds)
    # print(target_chunk)
    # trace_file_path = 'examples/testvector_gen4_m2_81p.pickle'
    # snn_model.save_trace_dict(trace_file_path)
    
    #################################################
    ## 1.test sample verification trace generation ##
    #################################################
    # import numpy as np
    # processed_data = np.array([test_ds[i][0] for i in range(len(test_ds))])
    # targets = np.array([test_ds[i][1] for i in range(len(test_ds))])
    
    # torch.save({
    #     'data': processed_data,
    #     'targets': targets
    # },
    #        'gen4cls_test500_v0.pth'    )
        
    