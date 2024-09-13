import os
import numpy as np

from eva import prophesee_parser as pp
from tonic import transforms
import matplotlib.pyplot as plt

label_id = {"0": "pedestrian", "1": "two wheeler", "2": "car", "3": "track", "4": "bus", "5": "traffic sign", "6": "traffic light"}

def parsing_data(path, save_path ,threshold = 500):
    idx = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            
            if file.endswith('.dat'):
                print("parsing file", file)
                dat_file = os.path.join(root, file)
                bbox_file = os.path.join(root, file.replace('td.dat', 'bbox.npy'))
                event_data = pp.PSEELoader(dat_file)
                
                
                bbox_data = np.load(bbox_file)
                
                print(f"total {len(bbox_data)} box parsing..")
                
                for bidx, bbox in enumerate(bbox_data):
                    ts, x, y, w, h, class_id, confidence, track_id = bbox
                    
                    x_int = int(x)
                    y_int = int(y)
                    w_int = int(w)
                    h_int = int(h)
                    
                    event_data.seek_time(ts)
                    event = event_data.load_delta_t(100_000)
                    
                    mask = (event["x"] >= x_int) & (event["x"] <(x_int+w_int)) & \
                    (event["y"]>=y_int) & (event["y"] <(y_int+h_int))
                    
                    event = event[mask]
                    event["x"] -= x_int
                    event["y"] -= y_int
                    
                    if (sum(event["p"] == 1) + sum(event["p"] == 0)) > threshold:
                        print(f"{bidx}/{len(bbox_data)} parsing event data / label {label_id[str(class_id)]}")
                        save_filepath = os.path.join(save_path, str(class_id))
                        os.makedirs(save_filepath, exist_ok=True)
                        np.save(os.path.join(save_filepath, str(idx)), event)
                        
                    else:
                        print(f"{bidx}/{len(bbox_data)} pass under {threshold} event count data / label {label_id[str(class_id)]}")
                        
                    idx+=1

if __name__ == "__main__":

    # path = '/data/Prophesee_Dataset_gen4_ad_mini/train'
    # save_path = '/data/Prophesee_Gen4_mini_cls/train'
    # os.makedirs(save_path, exist_ok=True)
    # print("training data converting...")
    # parsing_data(path, save_path)
    
    path = '/data/Prophesee_Dataset_gen4_ad_mini/val'
    save_path = '/data/Prophesee_Gen4_mini_cls/val'
    os.makedirs(save_path, exist_ok=True)
    print("validation data converting...")
    parsing_data(path, save_path)