
import pydantic
import torch
from typing import List
from dataclasses import dataclass
from typing import BinaryIO


class Vertex(pydantic.BaseModel):
    identifier: int
    x: int
    y: int
    area: float
    neighbours: List[int]

    def flip_xy(self) -> "Vertex":
        return Vertex(
            identifier=self.identifier,
            x=self.y,
            y=self.x,
            area=self.area,
            neighbours=self.neighbours)


@dataclass
class GridWorld:
    grid: torch.LongTensor


@dataclass
class TrainingSample:
    grid_world: GridWorld
    vertices: List[Vertex]

    @staticmethod
    def save_to_disk(samples: List["TrainingSample"], file_handle: BinaryIO) -> None:
        torch.save(samples, file_handle)

    @staticmethod
    def load_from_disk(file_handle: BinaryIO) -> List["TrainingSample"]:
        return torch.load(file_handle)
