from typing import Optional
from dataclasses import dataclass
import numpy as np
import tonic

@dataclass
class TemporalCrop:
    time_window: int = 99000

    def __call__(self, events):
        start_time = events['t'][0]
        end_time = start_time + self.time_window

        return events[(events["t"] >= start_time) & (events["t"] <= end_time)]
    

@dataclass
class RandomTemporalCrop:
    time_window: int = 99000 # default 99ms
    
    def __call__(self, events):
        start_time = np.random.randint(events['t'][0], events['t'][-1] - self.time_window)        
        end_time = start_time + self.time_window
        
        if (end_time - start_time) != self.time_window:
            raise IndexError('t index overflow')
        
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