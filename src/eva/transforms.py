from typing import Optional
from dataclasses import dataclass
import numpy as np
import tonic
from torchvision.transforms.functional import to_pil_image


@dataclass
class RandomTemporalCrop:
    time_window: int = 99000
    padding_enable: bool = False

    def __call__(self, events):
        if events['t'][-1] - events['t'][0] < self.time_window:
            if self.padding_enable:
                dummy_event = np.array([(events['t'][0] + self.time_window, 0, 0, 0)], dtype=events.dtype)
                events = np.concatenate((events, dummy_event))
                events = np.sort(events, order='t')

            else:
                raise ValueError("Time window is too small")
        
        if events['t'][0] == events['t'][-1] - self.time_window:
            start_time = events['t'][0]
        else:
            start_time = np.random.randint(events['t'][0], events['t'][-1] - self.time_window)
        end_time = start_time + self.time_window

        return events[(events["t"] >= start_time) & (events["t"] <= end_time)]

@dataclass
class TemporalCrop:
    time_window: int = 99000
    padding_enable: bool = False
    
    def __call__(self, events):
        start_time = events['t'][0]
        end_time = start_time + self.time_window
        
        if events['t'][-1] < self.time_window:
            if self.padding_enable:
                dummy_event = np.array([(events['t'][0] + self.time_window, 0, 0, 0)], dtype=events.dtype)
                events = np.concatenate((events, dummy_event))
                events = np.sort(events, order='t')
            else:
                raise ValueError("Time window is too small")
            
        return events[(events["t"] >= start_time) & (events["t"] <= end_time)]

@dataclass(frozen=True)
class ToFrameAuto:
    time_window: Optional[float] = None
    event_count: Optional[int] = None
    n_time_bins: Optional[int] = None
    n_event_bins: Optional[int] = None
    overlap: float = 0
    include_incomplete: bool = False

    def __call__(self, events):
        sensor_size = (max(events["x"]) + 1, max(events["y"]) + 1, 2)
        return tonic.transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=self.time_window,
            event_count=self.event_count,
            n_time_bins=self.n_time_bins,
            n_event_bins=self.n_event_bins,
            overlap=self.overlap,
            include_incomplete=self.include_incomplete,
        )(events) # for channel first format
        
@dataclass(frozen=True)
class MergeFramePolarity:
    bias: int = 128

    def __call__(self, frames):
        merged_frames = np.zeros((frames.shape[0],1,)+frames.shape[2:], dtype=np.int16)
    
        for i, frame in enumerate(frames):
            merged_frames[i][0] = self.bias + (frame[1] - frame[0])
        
        return merged_frames  # channel first format
        
@dataclass(frozen=True)
class EventFrameResize:
    size: tuple
    
    def __call__(self, frames):
        
        resized_frame = np.zeros(frames.shape[:2]+self.size, dtype=np.int16)
        for idx, frame in enumerate(frames):
            frame = frame.astype(np.uint8)
            pil_frame = to_pil_image(frame.transpose(1,2,0))
            resized_frame[idx] = pil_frame.resize(self.size)
        
        return resized_frame  # Stack frames back into a single tensor

@dataclass(frozen=True)
class EventNormalize:
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    
    def __call__(self, frames):
        return (frames - self.mean) / self.std
