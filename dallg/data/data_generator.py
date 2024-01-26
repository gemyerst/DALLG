import math
import numpy as np
import itertools
from typing import Dict
import torch
from .core import TrainingSample, Vertex, GridWorld


class RoomSpec:
    def __init__(self, r: int, c: int, size: int):
        self.r = r
        self.c = c
        self.left = c - size
        self.right = c + size
        self.top = r - size
        self.bottom = r + size
        self.size = size

    def overlaps_or_contains(self, other: "RoomSpec") -> bool:
        return (
                self.left <= other.right
                and self.right >= other.left
                and self.top <= other.bottom
                and self.bottom >= other.top
        )

    def adjacent_to(self, other: "RoomSpec") -> bool:
        return (
                self.overlaps_or_contains(other)
                or RoomSpec(self.r - 1, self.c, self.size).overlaps_or_contains(other)  # self shifted up
                or RoomSpec(self.r + 1, self.c, self.size).overlaps_or_contains(other)  # self shifted down
                or RoomSpec(self.r, self.c - 1, self.size).overlaps_or_contains(other)  # self shifted left
                or RoomSpec(self.r, self.c + 1, self.size).overlaps_or_contains(other)  # self shifted right
        )

    def completely_contains(self, other: "RoomSpec") -> bool:
        return (
                self.left <= other.left
                and self.right >= other.right
                and self.bottom >= other.bottom
                and self.top <= other.top
        )

    def contains_point(self, r: int, c: int) -> bool:
        return (
                self.left <= c <= self.right
                and self.top <= r <= self.bottom
        )


def generate_training_sample(target_n_rooms: int, grid_hw: int, strict_target=False) -> TrainingSample:
    # Generate :
    # array([21, 24, 48,  1, 26, 51,  0,  8, 38, 47])
    coords = np.random.choice(
        grid_hw - 6,               # so not too close to edge of grid
        size=target_n_rooms * 2,   # * 2 because (row, col)
    ) + 3                          # so not too close to edge of grid

    # Generate:
    # array([[47,  5],
    #        [11, 18],
    #        [21, 36],
    #        [52, 20],
    #        [14, 35]])
    rows_cols = coords.reshape(-1, 2)
    sizes = np.random.randint(low=1, high=2 * math.ceil(math.sqrt(grid_hw)), size=target_n_rooms)

    room_specs = list(sorted([
        RoomSpec(r, c, sizes[i])
        for (i, (r, c)) in enumerate(rows_cols)
    ], key=lambda spec: (spec.top, spec.left)))

    # Check if any Rooms completely contain the other
    superseded_spec_indices = set()
    for l in range(len(room_specs)):
        for r in range(l + 1, len(room_specs)):
            if room_specs[l].completely_contains(room_specs[r]):
                superseded_spec_indices.add(r)
    room_specs = [spec for (i, spec) in enumerate(room_specs) if i not in superseded_spec_indices]

    grid = np.full((grid_hw, grid_hw), -1)
    for r in range(grid_hw):
        for c in range(grid_hw):
            for i in range(len(room_specs)):
                if room_specs[i].contains_point(r, c):
                    grid[r, c] = i

    # # Ensure everyroom is at least 3 grids in width.    
    # grid = np.full((grid_hw, grid_hw), -1)
    # history = 0
    # for r in range(grid_hw):
    #     for c in range(grid_hw):
    #         if c % 3 == 0:
    #             history = grid[r, c]
    #         else:
    #         grid[r, c] = history 




    # Grid is now source of truth
    room_idxs, areas = np.unique(grid, return_counts=True)
    room2area: Dict[int, int] = {
        room_idxs[idx]: areas[idx]
        for (idx, room_idx) in enumerate(room_idxs)
        if room_idx >= 0
    }
    actual_n_rooms = len(room2area)
    if strict_target and actual_n_rooms != target_n_rooms:
        return generate_training_sample(target_n_rooms, grid_hw, strict_target)

    # See if we need to fill in any gaps
    if actual_n_rooms != target_n_rooms:
        new_room2area: Dict[int, int] = {}
        room_specs = [spec for (i, spec) in enumerate(room_specs) if i in room2area]
        for new_i, old_i in enumerate(sorted(room2area.keys())):
            new_room2area[new_i] = room2area[old_i]
            grid[grid == old_i] = new_i
        room2area = new_room2area

    vertices = [
        Vertex(identifier=i+1, x=room_specs[i].c, y=room_specs[i].r, area=room2area[i], neighbours=[])
        for i in range(actual_n_rooms)
    ]

    # Add adjacency info
    for (l, r) in itertools.combinations(room2area.keys(), r=2):
        if room_specs[l].adjacent_to(room_specs[r]):
            vertices[l].neighbours.append(r + 1)
            vertices[r].neighbours.append(l + 1)

    return TrainingSample(
        grid_world=GridWorld(grid=torch.LongTensor(grid + 1)),
        vertices=vertices)
