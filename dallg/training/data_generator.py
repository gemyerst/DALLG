
from threading import Event
from typing import Callable, TypeVar, List, Dict
from ..data.core import TrainingSample
from ..data.data_generator import generate_training_sample
from collections import defaultdict
import random


T = TypeVar('T')
U = TypeVar('U')
Function = Callable[[T], U]
Consumer = Callable[[T], None]


def run_generation_loop(kill: Event, data_preparer: Function[Event, T], data_saver: Consumer[T]):
    while True:
        kill.wait(timeout=0.0001)  # listen for kill signal
        data = data_preparer(kill)
        data_saver(data)


def default_data_preparer(kill: Event, target_n_samples: int, min_n_rooms: int, max_n_rooms: int, grid_hw: int) -> List[TrainingSample]:
    samples: Dict[int, List[TrainingSample]] = defaultdict(list)
    for _ in range(10_000):
        s = generate_training_sample(target_n_rooms=random.randint(1, 8), grid_hw=GRID_HW)
        samples[len(s.vertices)].append(s)
    # make sure equal number of samples of each n_rooms
    min_num = min(len(collection) for collection in samples.values())
    for k in samples.keys():
        samples[k] = samples[k][:min_num]
    return samples.values()
