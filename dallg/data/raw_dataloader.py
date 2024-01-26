
from typing import List, IO, Mapping, Any
from .core import Vertex, GridWorld, TrainingSample
import torch
import json



def load_vertices_for_sample(data_dict: Mapping[str, Any]) -> List[Vertex]:
    """
    Expected dataformat:
    {
        "vertexCount": <int>
        "0": {
            "x": <float, max=63>
            "y": <float, max=63>,
            "area": <float, max=64*64>,
            "roomType": <int, max=7>,
            "neighbours": [<int>, <int>, ...]
        }
    }
    """
    vertices = []
    vertex_count: int = data_dict["vertexCount"]
    for v in range(0, vertex_count + 0): # 0 for testing, 1 for training
        props = Vertex(**{
            **{"identifier": v + 1},
            **data_dict[str(v)],
        })
        vertices.append(props.flip_xy())
    return vertices


def load_sample(sample_file_handle: IO) -> TrainingSample:
    data_dict: Mapping[str, Any] = json.load(sample_file_handle)
    vertices = load_vertices_for_sample(data_dict)
    # Flip the grid 90 degrees to account for coordinate system difference
    grid = torch.reshape(torch.LongTensor(data_dict["roomIndex"]), (32, 32)).contiguous()
    grid = torch.rot90(grid, -1)
    return TrainingSample(grid_world=GridWorld(grid), vertices=vertices)
