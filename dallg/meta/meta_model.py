from ..graph.gcn_encoder import GCNEncoder
from ..persistence import PersistableModel, ModelPersister
from torch import FloatTensor, LongTensor, Tensor
from ..transformer import DecoderLayer, PositionalEncoding
from torch.nn import TransformerDecoderLayer
from torch import nn
import pydantic
import torch
from typing import Optional


class Config(pydantic.BaseModel):
    tf_tgt_vocab_size: int
    tf_d_model: int
    tf_num_enc_dec_heads: int
    tf_num_layers: int
    tf_d_ff: int
    tf_dropout: float
    gcn_n_node_features: int
    gcn_layer_hidden_dim: int
    gcn_n_out_features: int
    gcn_n_layers: int


def generate_no_peaking_target_mask(tgt: Tensor, rand_masking_pct: float) -> Tensor:
    """
    For each item in the target batch, generate a mask to prevent the decoder from peeking ahead.

    Returns a square matrix of shape (b, seq_len, seq_len) with True values,
    but whose upper triangular part is set to False. For e.g. a sequence length of 6:

        [[ True, False, False, False, False, False],
         [ True,  True, False, False, False, False],
         [ True,  True,  True, False, False, False],
         [ True,  True,  True,  True, False, False],
         [ True,  True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True,  True]]
    """
    (b, seq_length) = tgt.shape
    tri_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    rand_mask = torch.rand((b, seq_length, seq_length)) > rand_masking_pct
    return tri_mask & rand_mask


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GraphTransformerMetaModel(PersistableModel[Config]):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self._cfg = c
        self._gcn = GCNEncoder(n_node_features=c.gcn_n_node_features, n_out_features=c.gcn_n_out_features, n_layers=c.gcn_n_layers)
        self.decoder_embedding = nn.Embedding(c.tf_tgt_vocab_size + 2, c.tf_d_model)
        self.positional_encoding = PositionalEncoding(c.tf_d_model, 65)
        self.dropout = nn.Dropout(c.tf_dropout)
        self._tf = nn.ModuleList([
            TransformerDecoderLayer(d_model=c.tf_d_model, nhead=c.tf_num_enc_dec_heads, dim_feedforward=c.tf_d_ff, dropout=c.tf_dropout, batch_first=True)
            for _ in range(c.tf_num_layers)
        ])
        self._fc_out = nn.Linear(in_features=c.tf_d_model, out_features=c.tf_tgt_vocab_size + 2)
        self.apply(_init_weights)
    
    def config(self) -> Config:
        return self._cfg

    def forward(self, codebooks: LongTensor, adj_matrices: FloatTensor, v_features: FloatTensor, rand_masking_pct: float, device: torch.device) -> Tensor:
        graph_logits = self.encode(adj_matrices=adj_matrices, v_features=v_features)       # (b, n_nodes, d_model)
        return self.decode(codebooks=codebooks, memory=graph_logits, rand_masking_pct=rand_masking_pct, device=device)

    def encode(self, adj_matrices: FloatTensor, v_features: FloatTensor) -> FloatTensor:
        graph_logits = self._gcn(node_features=v_features, adj_matrix=adj_matrices)  # (b, n_vertices, d_model)
        return graph_logits

    def decode(self, codebooks: LongTensor, memory: Tensor, device: torch.device, rand_masking_pct: Optional[float] = None) -> Tensor:
        if rand_masking_pct is not None:
            tgt_mask = generate_no_peaking_target_mask(tgt=codebooks, rand_masking_pct=rand_masking_pct).to(device)
        else:
            tgt_mask = None

        tgt = self.dropout(self.positional_encoding(self.decoder_embedding(codebooks)))
        for layer in self._tf:
            tgt = layer(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_is_causal=rand_masking_pct is not None)

        return self._fc_out(tgt)


class GraphTransformerMetaModelPersister(ModelPersister[GraphTransformerMetaModel, Config]):
    
    def instantiate_model(self, config: Config) -> GraphTransformerMetaModel:
        return GraphTransformerMetaModel(c=config)
