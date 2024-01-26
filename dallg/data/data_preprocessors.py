
from typing import List, Tuple, Callable, Iterable, Mapping
import torch
from torch import LongTensor, Tensor
import torch.nn.functional as F
from .raw_dataloader import TrainingSample
from .core import Vertex
from random import shuffle
from collections import defaultdict
from torch.utils.data import random_split


class GraphDatasetItem:
    n_vertices: int
    tensors: Tuple[Tensor, ...]

    def __init__(self, n_vertices: int, *tensors: Tensor):
        self.n_vertices = n_vertices
        self.tensors = tensors

    def prepend(self, new_tensor: Tensor) -> "GraphDatasetItem":
        return GraphDatasetItem(self.n_vertices, new_tensor, *self.tensors)


class GraphDatasetBatch:
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    @staticmethod
    def from_items(items: List[GraphDatasetItem]) -> "GraphDatasetBatch":
        num_items = len(items[0].tensors)
        return GraphDatasetBatch(*[
            torch.cat([item.tensors[col].unsqueeze(0) for item in items])
            for col in range(num_items)
        ])


class GraphDatasetIterableFactory:
    def __init__(self, batches: List[GraphDatasetBatch]):
        self._batches = batches

    def new_iterable(self) -> Iterable[GraphDatasetBatch]:
        indexes = list(range(len(self._batches)))
        shuffle(indexes)
        for i in indexes:
            yield self._batches[i]

    def __len__(self):
        return len(self._batches)

    def split(self, retain_pct: float) -> Tuple["GraphDatasetIterableFactory", "GraphDatasetIterableFactory"]:
        (retain, give) = random_split(self._batches, lengths=[retain_pct, 1 - retain_pct])
        return GraphDatasetIterableFactory(retain), GraphDatasetIterableFactory(give)


class GraphDataset:
    buckets: Mapping[int, List[GraphDatasetItem]]

    def __init__(self, buckets: Mapping[int, List[GraphDatasetItem]]):
        self.buckets = buckets

    def create_iterator_factory(self, batch_size: int) -> GraphDatasetIterableFactory:
        all_batches: List[GraphDatasetBatch] = []
        batch: List[GraphDatasetItem] = []
        for (_, segment) in sorted(self.buckets.items(), key=lambda kv: kv[0]):
            for item in segment:
                if batch and (len(batch) >= batch_size or batch[-1].n_vertices != item.n_vertices):
                    all_batches.append(GraphDatasetBatch.from_items(batch))
                    batch = [item]
                else:
                    batch.append(item)
        if batch:
            all_batches.append(GraphDatasetBatch.from_items(batch))
        return GraphDatasetIterableFactory(batches=all_batches)


def preprocess_prediction(vertices: List[Vertex]) -> GraphDataset:
    buckets = defaultdict(list)

    n_vertices = len(vertices)
    adj = torch.zeros((n_vertices, n_vertices))
    features = torch.zeros((n_vertices, 3))

    for vertex in sorted(vertices, key=lambda v: v.x + v.y):
        v_id = vertex.identifier - 1
        # update features
        features[v_id][0] = float(vertex.x) / 64
        features[v_id][1] = float(vertex.y) / 64
        features[v_id][2] = float(vertex.area) / 4096  # 64 * 64
        # update adjacency matrix
        adj[v_id][v_id] = 1.0
        for neighbour_id in vertex.neighbours:
            adj[v_id][neighbour_id - 1] = 1.0
            adj[neighbour_id - 1][v_id] = 1.0

    buckets[n_vertices].append(GraphDatasetItem(n_vertices, adj, features))
    return GraphDataset(buckets=buckets)


def preprocess_vae(samples: List[TrainingSample], max_vertices: int) -> GraphDataset:
    buckets = defaultdict(list)

    for sample in samples:
        n_vertices = len(sample.vertices)

        grid = F.one_hot(sample.grid_world.grid, num_classes=max_vertices).permute(2, 0, 1).float()
        adj = torch.zeros((n_vertices, n_vertices))
        features = torch.zeros((n_vertices, 3))

        for vertex in sorted(sample.vertices, key=lambda v: v.x + v.y):
            v_id = vertex.identifier - 1
            # update features
            features[v_id][0] = float(vertex.x) / 64
            features[v_id][1] = float(vertex.y) / 64
            features[v_id][2] = float(vertex.area) / 4096  # 64 * 64
            # update adjacency matrix
            adj[v_id][v_id] = 1.0
            for neighbour_id in vertex.neighbours:
                adj[v_id][neighbour_id - 1] = 1.0
                adj[neighbour_id - 1][v_id] = 1.0

        buckets[n_vertices].append(GraphDatasetItem(n_vertices, grid, adj, features))

    return GraphDataset(buckets=buckets)


def preprocess_transformer(samples: List[TrainingSample], codebook_encoder: Callable[[LongTensor], LongTensor], max_vertices: int) -> GraphDataset:
    vae_dataset = preprocess_vae(samples=samples, max_vertices=max_vertices)

    # convert grids to codebook indices
    for (i, segment) in vae_dataset.buckets.items():
        grids = torch.cat([item.tensors[0].unsqueeze(0) for item in segment])   # (N, max_vertices, 64, 64)
        codebooks = codebook_encoder(grids)     # (N, 64)
        for t in range(len(segment)):
            item = segment[t]
            book = codebooks[t]
            segment[t] = item.prepend(book)

    return vae_dataset
