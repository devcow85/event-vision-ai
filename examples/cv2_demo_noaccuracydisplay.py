import cv2
import numpy as np
import tonic.transforms as transforms
import torch

from eva import transforms as eva_transforms
from eva import prophesee_parser as pp
from scipy.stats import linregress

from eva import model

def iou(box1, box2):
    # 각 BBOX의 좌표 계산
    x1_min, y1_min = box1[1], box1[2]
    x1_max, y1_max = box1[1] + box1[3], box1[2] + box1[4]

    x2_min, y2_min = box2[1], box2[2]
    x2_max, y2_max = box2[1] + box2[3], box2[2] + box2[4]

    # 교집합 좌표 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 교집합 면적 계산
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 각 BBOX의 면적 계산
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 합집합 면적 계산
    union_area = box1_area + box2_area - inter_area

    # IoU 계산
    return inter_area / union_area if union_area != 0 else 0

def merge_boxes(box1, box2):
    # 두 박스를 합쳐서 새로운 박스로 생성
    x_min = min(box1[1], box2[1])
    y_min = min(box1[2], box2[2])
    width = max(box1[1] + box1[3], box2[1] + box2[3]) - x_min
    height = max(box1[2] + box1[4], box2[2] + box2[4]) - y_min

    # 새로운 BBOX 생성, 다른 값은 첫 번째 박스의 값을 유지
    return (box1[0], x_min, y_min, width, height, box1[5], box1[6], box1[7])

def remove_overlapping_bboxes(bboxes, iou_threshold=0.5):
    bboxes = np.array(bboxes)
    keep_boxes = []

    while len(bboxes) > 0:
        # 첫 번째 박스를 기준으로 IoU 계산
        base_box = bboxes[0]
        keep_boxes.append(base_box)
        bboxes = np.delete(bboxes, 0, axis=0)
        to_delete = []

        # 남은 박스들에 대해 IoU 계산
        for i, box in enumerate(bboxes):
            if iou(base_box, box) > iou_threshold:
                # 중복되면 박스를 합침
                keep_boxes[-1] = merge_boxes(keep_boxes[-1], box)
                to_delete.append(i)

        # 중복된 박스를 제거
        bboxes = np.delete(bboxes, to_delete, axis=0)

    return keep_boxes

# def crop_event_data(event_data, box, padding=10):
#     """ 바운딩 박스에 해당하는 이벤트 데이터를 크롭 """
#     x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
#     x_min = max(0, x - padding)
#     y_min = max(0, y - padding)
#     x_max = x + w + padding
#     y_max = y + h + padding
#     cropped_event_data = event_data[(event_data["x"] >= x_min) & (event_data["x"] <= x_max) &
#                                     (event_data["y"] >= y_min) & (event_data["y"] <= y_max)]
#     cropped_event_data["x"] -= int(x_min)
#     cropped_event_data["y"] -= int(y_min)
    
#     return cropped_event_data

def crop_event_data(event_data, box, padding=10):
    """ 바운딩 박스에 해당하는 이벤트 데이터를 1:1 비율로 크롭 """
    x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    
    # w와 h 중 큰 값을 기준으로 1:1 비율로 조정
    max_side = max(w, h)
    
    # 너비와 높이를 1:1 비율로 맞추기 위해 각각 padding 적용
    if w < max_side:
        # w가 h보다 작으면 좌우로 패딩
        x_min = max(0, x - (max_side - w) // 2 - padding)
        x_max = x + w + (max_side - w) // 2 + padding
    else:
        # h가 w보다 작으면 상하로 패딩
        x_min = max(0, x - padding)
        x_max = x + w + padding

    if h < max_side:
        # h가 w보다 작으면 상하로 패딩
        y_min = max(0, y - (max_side - h) // 2 - padding)
        y_max = y + h + (max_side - h) // 2 + padding
    else:
        # w가 h보다 작으면 좌우로 패딩
        y_min = max(0, y - padding)
        y_max = y + h + padding

    # 이벤트 데이터를 크롭
    cropped_event_data = event_data[(event_data["x"] >= x_min) & (event_data["x"] <= x_max) &
                                    (event_data["y"] >= y_min) & (event_data["y"] <= y_max)]
    
    # 크롭된 이벤트 데이터의 좌표를 원점으로 이동
    cropped_event_data["x"] -= int(x_min)
    cropped_event_data["y"] -= int(y_min)

    return cropped_event_data


def create_grid_image(images, grid_size=(3, 3), image_size=(64, 64)):
    """ 격자형 이미지 생성 및 격자선 그리기 """
    rows, cols = grid_size
    img_h, img_w = image_size
    grid_img = np.zeros((img_h * rows, img_w * cols, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid_img[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = img
    
    # 격자선 그리기
    for row in range(1, rows):
        cv2.line(grid_img, (0, row * img_h), (cols * img_w, row * img_h), (255, 255, 255), 1)
    for col in range(1, cols):
        cv2.line(grid_img, (col * img_w, 0), (col * img_w, rows * img_h), (255, 255, 255), 1)
    
    return grid_img

def split_t_into_bins(event_data, n_bins=5):
    """
    주어진 데이터에서 t 값을 n개의 구간으로 나누고, 각 구간에 속하는 이벤트의 개수를 출력하는 함수.
    """
    # t 값의 최소와 최대 구하기
    t_min = event_data['t'].min()
    t_max = event_data['t'].max()

    # t 값의 구간을 n_bins개의 구간으로 나눔
    bins = np.linspace(t_min, t_max, n_bins + 1)

    # 각 구간에 해당하는 이벤트 개수 계산
    bin_counts = np.histogram(event_data['t'], bins=bins)[0]

    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_centers, bin_counts

def calculate_linear_regression(bin_centers, bin_counts):
    """
    주어진 구간 중앙값 (bin_centers)와 이벤트 개수 (bin_counts)에 대해 선형 회귀를 적용해 기울기를 계산.
    """
    slope, intercept, r_value, p_value, std_err = linregress(bin_centers, bin_counts)
    return slope

label_map = {"0": "pedestrian", "1": "two wheeler", "2": "car", "3": "truck", "4": "bus", "5": "traffic sign", "6": "traffic light"}

label_map_snn = {"0": "bus", "1": "car", "2": "pedestrian", "3": "truck", "4": "two_wheeler"}

class_colors = {
    0: (255, 0, 0),      # pedestrian (빨간색)
    1: (0, 255, 0),      # two wheeler (초록색)
    2: (0, 0, 255),      # car (파란색)
    3: (255, 255, 0),    # truck (노란색)
    4: (255, 0, 255),    # bus (분홍색)
    5: (0, 255, 255),    # traffic sign (청록색)
    6: (128, 0, 128)     # traffic light (보라색)
}

if __name__ == "__main__":
    dat_file = "/data/event-data/Prophesee_Dataset_gen4_ad_mini/val/moorea_2019-01-30_000_td_549500000_609500000_td.dat"
    event_stream = pp.PSEELoader(dat_file)
    npy_file = dat_file.replace('td.dat', 'bbox.npy')
    bbox = np.load(npy_file)
    
    i = 0           # event_frame buffer
    step = 33_000   # unit us
    num_buffer = 2
    event_buffer = []

    transform = transforms.Compose([
        transforms.ToFrame((1280, 720, 2), n_time_bins=1),
    ])
    
    preprocessing = transforms.Compose([
        eva_transforms.ToFrameAuto(n_time_bins=5),
        eva_transforms.MergeFramePolarity(),
        eva_transforms.EventFrameResize((64,64)),
        eva_transforms.EventNormalize(mean=(128,), std=(1,))
    ])
    
    snn_model = model.PPGen4NetMini(5, 1, 5, "result/PPGen4NetMini_0/best_model.pt")
    snn_model.cuda()
    
    total_len = 0
    match_count = 0

    while True:
        cur_time = i * step

        # extract bbox
        bbox_idx = (bbox['ts'] >= cur_time) & (bbox['ts'] <= cur_time + step * len(event_buffer))
        cur_bbox = bbox[bbox_idx]
        cur_bbox = [box for box in cur_bbox if int(box[5]) not in [5, 6]]

        # remove overlapping bbox
        filtered_bbox = remove_overlapping_bboxes(cur_bbox, iou_threshold=0.5)

        # extract event_data for unit time length
        event_stream.seek_time(cur_time)
        event_data = event_stream.load_delta_t(step)
        i+=1
        
        # save event-data to buffer
        event_buffer.append(event_data)
        
        # concat buffer to one event stream for inference
        ceb = np.concatenate(event_buffer)
        
        # preprocess for visualization 
        event_frame = transform(ceb)[0]
        event_frame = (event_frame[1] - event_frame[0]).astype(np.uint8)

        img = cv2.cvtColor(event_frame * 255, cv2.COLOR_GRAY2BGR)

        cropped_images = []
        snn_inputs = []
        gt_labels = []
        # draw bbox and class_id
        for idx, box in enumerate(filtered_bbox):
            total_len+=1
            x, y, w, h, class_id = int(box[1]), int(box[2]), int(box[3]), int(box[4]), box[5]
            
            # class_id에 따른 색상 선택
            color = class_colors.get(class_id, (255, 255, 255))  # 기본값: 흰색
            
            # draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            label = label_map.get(str(class_id), "Unknown")
            
            # add class_id text
            text_position = (x, y - 10)  # place text above the bounding box
            cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cropped_event_data = crop_event_data(ceb, box, padding=1)
            
            bin_centers, bin_counts = split_t_into_bins(ceb)
            
            cropped_frame = preprocessing(cropped_event_data)
            
            snn_input_tensor = torch.tensor(cropped_frame).to(torch.float)
            snn_input_tensor = snn_input_tensor.unsqueeze(0)
            output = snn_model(snn_input_tensor.cuda())
            output_spike = torch.sum(output, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            pred = label_map_snn[str(output_spike[0].argmax())]
            
            if label==pred:
                match_count+=1 
                
            snn_inputs.append(cropped_frame)
            
            cropped_img = cropped_frame.sum(axis=0)[0]
            cropped_img = cv2.cvtColor(cropped_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_labels.append(label)
            
            cv2.putText(cropped_img, str(class_id), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(cropped_img, pred, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # cv2.putText(cropped_img, f"{std_bin/1000:.1f}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cropped_images.append(cropped_img)

        if cropped_images:
            grid_img = create_grid_image(cropped_images, grid_size=(5,5), image_size=(64, 64))
            cv2.namedWindow("Grid Image", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Grid Image", grid_img)

        cv2.putText(img, f"event_buffer_count: {i}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(img, f"time: {cur_time/1e6:.2f}sec", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(img, f"# of objects: {len(filtered_bbox)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        # if total_len > 0:
        #     cv2.putText(img, f"Accuracy: {match_count/total_len*100:.2f}%", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        #     cv2.putText(img, f"Match/Total: {match_count}/{total_len}", (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        cv2.imshow('Event Frame', img)
        
        # fifo for 66ms event buffer
        if len(event_buffer) >= num_buffer:
            event_buffer.pop(0)
        
        # check final event
        if event_stream.current_time > event_stream.total_time():
            # break
            i = 0 # reset
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        

cv2.destroyAllWindows()
