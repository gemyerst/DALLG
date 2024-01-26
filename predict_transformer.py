
from config import *
from dallg.meta.meta_model import GraphTransformerMetaModelPersister, GraphTransformerMetaModel
from dallg.data.data_preprocessors import preprocess_prediction
from dallg.data.core import TrainingSample, Vertex
from dallg.data.data_generator import generate_training_sample
from dallg.data.raw_dataloader import load_vertices_for_sample
from torch.utils.data import DataLoader, TensorDataset
from dallg.vae.vae import DiscreteVAE
from dallg.vae.vae_persister import VaePersister
from typing import cast, List
import torch
import traceback


from flask import Flask, request, jsonify
app = Flask(__name__)


PRETRAINED_VAE: DiscreteVAE
META_MODEL: GraphTransformerMetaModel
vae_weights_path = "weights/vae_softmax_v4_190_0.01543_0.995.pt"
tf_weights_path = "weights/tf_softmax_v4_180_0.16558_0.933.pt"


def predict(vertices: List[Vertex]) -> torch.LongTensor:
    """
    Predict on a training sample, and return a 32x32 grid
    """
    global META_MODEL
    global PRETRAINED_VAE

    dataset = preprocess_prediction(vertices=vertices)

    with torch.no_grad():
        for batch in dataset.create_iterator_factory(batch_size=1).new_iterable():
            batch_accelerated = [x.to(DEVICE_ACCELERATION) for x in batch.tensors]
            (adj_matrices, v_features) = batch_accelerated
            (b, _, _) = adj_matrices.shape

            # Get just first token
            tgt = torch.zeros((b, 1), dtype=torch.long).to(DEVICE_ACCELERATION)
            tgt[:, 0] = VAE_CODEBOOK_TOKENS_VOCAB_SIZE  # <SOS> token

            memory = META_MODEL.encode(adj_matrices, v_features)

            for _ in range(65):
                prediction = META_MODEL.decode(codebooks=tgt, memory=memory, device=DEVICE_ACCELERATION, rand_masking_pct=None)     # (b, n_nodes, vocab_size)
                prediction = prediction[:, -1, :]                                                                                   # (b, vocab_size)
                predicted_ids = torch.argmax(prediction, axis=-1)                                                                   # (b,)
                tgt = torch.cat([tgt, predicted_ids.unsqueeze(1)], dim=1)

            squished_grid = PRETRAINED_VAE.decode(tgt[:, 1:-1]).squeeze(0).argmax(dim=0)    # (32, 32)
            return squished_grid


def initialize_models():
    global PRETRAINED_VAE
    global META_MODEL

    # Load vae from persisted
    with open(vae_weights_path, "rb") as file:
        PRETRAINED_VAE, _ = VaePersister().restore(file)
        PRETRAINED_VAE = PRETRAINED_VAE.to(DEVICE_ACCELERATION).eval()
    assert PRETRAINED_VAE is not None
    
    # Load from persisted
    persister = GraphTransformerMetaModelPersister()
    with open(tf_weights_path, "rb") as file:
        META_MODEL, _ = persister.restore(io_handle=file)
        META_MODEL = META_MODEL.to(DEVICE_ACCELERATION).eval()
    assert META_MODEL is not None



@app.route("/predict", methods=["POST"])
def prediction_route():
    try:
        content = request.get_json(force=True)
        vertices: List[Vertex] = load_vertices_for_sample(content)
        prediction = predict(vertices=vertices)
        return jsonify({
            "Prediction": prediction.reshape(-1).tolist()
        })
    except Exception as e:
        print(e)
        traceback.print_exc()
        

if __name__ == "__main__":
    initialize_models()
    app.run(host='0.0.0.0', port=8080, debug=True)
