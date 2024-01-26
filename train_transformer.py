import os
from dallg.data.data_generator import generate_training_sample, TrainingSample
from dallg.data.raw_dataloader import load_sample
import uuid
import time
from config import *
import random
import shutil
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import List

# PARAMETERS
low_rooms = 1
high_rooms = 10
max_workers = 12
load_existing_model = True
vae_weights_path = "weights/vae_softmax_finetune_370_0.00963_0.997.pt"
tf_weights_path = "./tf_softmax_fromScratch_6000_0.3177_0.872.pt"
starting_epoch = 11901
data_generation_target = 20_000
real_data_pct = 0.0
data_reduction_by_room = 1


def endlessly_generate_samples():
    save_dir = "./temp_write"
    dest_dir = "./temp_read"
    indices = list(range(80_000))

    def shuffle_list(l: List) -> List:
        random.shuffle(l)
        return l

    while True:
        samples = defaultdict(list)
        # Select samples from the JSON data
        random.shuffle(indices)
        json_indices_to_select = indices[:int(real_data_pct * data_generation_target)]
        for idx in json_indices_to_select:
            try:
                with open(os.path.join(r"C:\Users\georgina.myers\gh_scripts\32Grid_data", f"g_{idx}.json"), "rb") as json_file:
                    json_sample = load_sample(json_file)
                    tray = samples[len(json_sample.vertices)]
                    tray.append(json_sample)
            except FileNotFoundError as e:
                # Ignore missing files
                continue
            except Exception as e:
                traceback.print_exc()
                raise e
        # Mix in the generated data samples
        for _ in range(int((1 - real_data_pct) * data_generation_target)):
            
            s = generate_training_sample(target_n_rooms=random.randint(low_rooms + 2, high_rooms + 4), grid_hw=GRID_HW)

            tray = samples[len(s.vertices)]
            tray.append(s)
        # Clip to correct n_rooms, and shuffle the ordering of generated & real data
        samples = {k: shuffle_list(tray) for k, tray in samples.items() if k <= high_rooms}
        # Rebalance the dataset to favour higher numbers of rooms
        high_rooms_count = len(samples[high_rooms])
        k = 0
        for i in range(high_rooms, 0, -1):  # 8, 7, 6, 5, 4, 3, 2, 1
            num_to_keep = max(int(data_reduction_by_room**k * high_rooms_count), 100)
            samples[i] = samples[i][:num_to_keep]
            k += 1
        # Print stats
        if random.random() < 0.2:
            key_tray_tuples = [(k, len(tray)) for k, tray in samples.items()]
            print("data distribution", sorted(key_tray_tuples, key=lambda kv: kv[0]))
        # Attempt to save samples to file system
        while True:
            if len(os.listdir(dest_dir)) < 10:
                fname = f"{uuid.uuid4()}.pt"
                with open(os.path.join(save_dir, fname), "wb") as f:
                    TrainingSample.save_to_disk(samples=[s for collection in samples.values() for s in collection], file_handle=f)
                shutil.move(os.path.join(save_dir, fname), os.path.join(dest_dir, fname))
                break
            else:
                time.sleep(random.randint(5, 15))


def main():
    import os
    from dallg.data.data_generator import TrainingSample
    import time
    import torch
    from typing import cast
    from torch.utils.data import DataLoader, TensorDataset
    from dallg.vae.vae_persister import VaePersister, DiscreteVAE
    from dallg.data.data_preprocessors import preprocess_transformer, GraphDataset, GraphDatasetBatch
    from torch.optim import Adam
    from torch import nn
    from dallg.meta.meta_model import GraphTransformerMetaModel, Config, GraphTransformerMetaModelPersister
    from concurrent.futures import ProcessPoolExecutor

    pretrained_vae: DiscreteVAE
    with open(vae_weights_path, "rb") as file:
        pretrained_vae, _ = VaePersister().restore(file)
        pretrained_vae = pretrained_vae.eval()
    assert pretrained_vae is not None

    def load_next_dataset_from_disk() -> GraphDataset:
        files = os.listdir("./temp_read")
        if not files:
            sleep_sec = 5
            print(f"no files yet, waiting {sleep_sec} seconds before re-checking")
            time.sleep(sleep_sec)
            return load_next_dataset_from_disk()

        path = os.path.join("./temp_read", files[0])
        with open(path, "rb") as f:
            samples = TrainingSample.load_from_disk(f)
        os.remove(path)

        def encoder_fn(grids: torch.LongTensor) -> torch.LongTensor:
            (N, _, _, _) = grids.shape
            processed = []

            pretrained_vae.to(DEVICE_ACCELERATION)
            for (grid,) in DataLoader(TensorDataset(grids), batch_size=128):
                encoded = pretrained_vae.get_codebook_indices(grid.to(DEVICE_ACCELERATION)).cpu()
                (b, _) = encoded.shape
                sos_tokens = torch.full((b, 1), VAE_CODEBOOK_TOKENS_VOCAB_SIZE)
                eos_tokens = torch.full((b, 1), VAE_CODEBOOK_TOKENS_VOCAB_SIZE + 1)
                processed.append(torch.cat([sos_tokens, encoded, eos_tokens], dim=1))
            pretrained_vae.cpu()
            return cast(torch.LongTensor, torch.cat(processed))

        return preprocess_transformer(samples, max_vertices=MAX_VERTICES, codebook_encoder=encoder_fn)
    
    # Train From Scratch
    if not load_existing_model:
        meta_model = GraphTransformerMetaModel(Config(
            tf_tgt_vocab_size=VAE_CODEBOOK_TOKENS_VOCAB_SIZE,
            tf_d_model=TF_MODEL_DIMS,
            tf_num_enc_dec_heads=TF_N_ENC_DEC_HEADS,
            tf_num_layers=TF_N_LAYERS,
            tf_d_ff=TF_FF_DIMS,
            tf_dropout=TF_DROPOUT,
            gcn_n_node_features=GCN_N_NODE_FEATURES,
            gcn_layer_hidden_dim=GCN_LAYER_HIDDEN_DIM,
            gcn_n_out_features=TF_MODEL_DIMS,
            gcn_n_layers=GCN_N_LAYERS)
        ).to(DEVICE_ACCELERATION)
        meta_model_persister = GraphTransformerMetaModelPersister()
        optimizer = Adam(meta_model.parameters(), lr=TF_LEARNING_RATE)
    else:
    # Load existing model
        meta_model_persister = GraphTransformerMetaModelPersister()
        with open(tf_weights_path, "rb") as f:
            meta_model, optimizer_state_dict = meta_model_persister.restore(f)
            meta_model.to(DEVICE_ACCELERATION)
            optimizer = Adam(meta_model.parameters(), lr=TF_LEARNING_RATE)
            # if optimizer_state_dict:
            #     optimizer.load_state_dict(optimizer_state_dict)

    criterion = nn.CrossEntropyLoss()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        [pool.submit(endlessly_generate_samples) for _ in range(max_workers)]

        # Run training loop
        for epoch in tqdm(range(starting_epoch, 15000)):
            epoch_loss = 0
            n_batches = 0
            n_examples = defaultdict(int)
            epoch_correct = defaultdict(int)

            dataset = load_next_dataset_from_disk()
            train_dataloader = dataset.create_iterator_factory(batch_size=int(os.getenv("BATCH_SIZE")))
            clip_value = 0.01
            for (i, batch) in enumerate(train_dataloader.new_iterable()):
                batch_accelerated = [x.to(DEVICE_ACCELERATION) for x in batch.tensors]
                (codebook, _, adj_matrices, v_features) = batch_accelerated
                (batch_size, n_rooms, _) = v_features.shape

                # Forward pass. Note - codebooks content is [<SOS>, x1, x2, ... <EOS>]
                decoder_input = codebook[:, :-1]  # everything up to <EOS>, shape=(b, 64 + 1)
                decoder_target = codebook[:, 1:]  # everything after <SOS>, shape=(b, 64 + 1)
                tf_logits = meta_model(decoder_input, adj_matrices, v_features, 0.001, DEVICE_ACCELERATION)  # (b, 64 + 1, vocab_size)

                # compute loss
                loss_input = tf_logits.reshape(-1, tf_logits.shape[-1])
                loss_target = decoder_target.reshape(-1)
                loss = criterion(loss_input, loss_target)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_model.parameters(), clip_value) # initial epochs 1.0, later epochs 0.05
                optimizer.step()

                # update statistics
                epoch_loss += loss.item()
                n_examples[n_rooms] += batch_size
                epoch_correct[n_rooms] += torch.eq(decoder_target, tf_logits.argmax(dim=-1)).sum().item()
                n_batches += 1

            # Log stats
            mean_loss = round(epoch_loss / n_batches, 5)
            mean_accuracy = round(sum(epoch_correct.values()) / (sum(n_examples.values()) * (64+1)), 3)
            accuracies_per_room = []
            for n in range(low_rooms, high_rooms + 1):
                if n in epoch_correct:
                    pct = epoch_correct[n] / (n_examples[n] * (64+1))
                    accuracies_per_room.append((n, round(pct, 3)))
            print(f"epoch: {epoch}, mean_loss: {mean_loss}, overall_accuracy: {mean_accuracy}, accuracies_per_room: {accuracies_per_room}")

            if epoch % 100 == 0:
                # Persist Model checkpoint and optimizer
                with open(f"tf_softmax_fromScratch_{epoch}_{mean_loss}_{mean_accuracy}.pt", "wb") as outfile:
                    meta_model_persister.persist_model(model=meta_model, io_handle=outfile, optimizer=optimizer)

        sys.exit(0)

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)
