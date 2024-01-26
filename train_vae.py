
import os
from dallg.data.data_generator import generate_training_sample, TrainingSample
from dallg.data.raw_dataloader import load_sample
import uuid
import time
from config import *
import random
import shutil
import sys
from collections import defaultdict
from tqdm import tqdm
from dallg.data.data_preprocessors import preprocess_vae

load_existing_model = True
vae_weights_path = "./weights/vae_softmax_210_0.01791_0.994.pt"

low_rooms = 1
high_rooms = 10
data_generation_target = 20_000
real_data_pct = 0.3
data_reduction_by_room = 0.9


def endlessly_generate_samples():
    save_dir = "./temp_write"
    dest_dir = "./temp_read"
    indices = list(range(80_000))

    def shuffle_list(l: list) -> list:
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
    from dallg.vae.vae_persister import VaePersister, DiscreteVAE
    from dallg.data.data_preprocessors import GraphDataset
    from torch.optim import Adam
    from dallg.vae.vae import Config
    from concurrent.futures import ProcessPoolExecutor

    # Train From Scratch
    if not load_existing_model:
        vae = DiscreteVAE(c=Config(
            image_size=GRID_HW,
            num_tokens=VAE_CODEBOOK_TOKENS_VOCAB_SIZE,
            codebook_dim=VAE_CODEBOOK_DIM,
            num_classes=MAX_VERTICES,
            num_layers=VAE_N_LAYERS,
            num_resnet_blocks=VAE_N_RESNET_BLOCKS,
            hidden_dim=VAE_HIDDEN_DIM,
            channels=VAE_N_CHANNELS,
            smooth_l1_loss=VAE_SMOOTH_L1_LOSS,
            temperature=VAE_TEMP_STARTING_VALUE,
            straight_through=VAE_STRAIGHT_THROUGH,
            reinmax=VAE_REINMAX,
            kl_div_loss_weight=VAE_KL_DIV_LOSS_WEIGHT)
        ).to(DEVICE_ACCELERATION)
        optimizer = Adam(vae.parameters(), lr=VAE_LEARNING_RATE)

    else:
    # Load existing model
        vae: DiscreteVAE
        with open(vae_weights_path, "rb") as file:
            vae, optimizer_state_dict = VaePersister().restore(file)
            vae = vae.to(DEVICE_ACCELERATION)
            optimizer = Adam(vae.parameters(), lr=VAE_LEARNING_RATE)
            if optimizer_state_dict is not None:
                optimizer.load_state_dict(optimizer_state_dict)

    def load_next_dataset_from_disk() -> GraphDataset:
        files = os.listdir("./temp_read")
        if not files:
            sleep_sec = 10
            print(f"no files yet, waiting {sleep_sec} seconds before re-checking")
            time.sleep(sleep_sec)
            return load_next_dataset_from_disk()
        path = os.path.join("./temp_read", files[0])
        with open(path, "rb") as f:
            samples = TrainingSample.load_from_disk(f)
        os.remove(path)
        return preprocess_vae(samples, max_vertices=MAX_VERTICES)

    with ProcessPoolExecutor(max_workers=8) as pool:
        [pool.submit(endlessly_generate_samples) for _ in range(8)]

        # Run training loop
        for epoch in tqdm(range(1, 1000)):
            epoch_loss = 0
            n_batches = 0
            n_examples = defaultdict(int)
            epoch_correct = defaultdict(int)

            dataset = load_next_dataset_from_disk()
            train_dataloader = dataset.create_iterator_factory(batch_size=VAE_BATCH_SIZE)

            total = len(train_dataloader)
            for (i, batch) in enumerate(train_dataloader.new_iterable()):
                grid, adj, features = batch.tensors
                (batch_size, n_rooms, _) = features.shape  # b, n_rooms, 64, 64
                grid = grid.to(DEVICE_ACCELERATION)

                # Forward through VAE
                loss, recons = vae(grid, return_loss=True, return_recons=True, temp=VAE_TEMP_STARTING_VALUE)

                # Update gradients
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 0.001)
                optimizer.step()

                # record the loss
                epoch_loss += loss.item()
                # record no. of correct indices (for computing accuracy later)
                reconstruction = recons.argmax(dim=1)       # (b, n_rooms, 64, 64) -> (b, 64, 64)
                original = grid.argmax(dim=1)               # (b, n_rooms, 64, 64) -> (b, 64, 64)
                epoch_correct[n_rooms] += torch.eq(original, reconstruction).sum().item()
                # increment number of batches and examples seen
                n_batches += 1
                n_examples[n_rooms] += batch_size
                # update progress bar
                print(f"batch={i}/{total}, loss={round(loss.item(), 3)} accuracy={round(sum(epoch_correct.values()) / (sum(n_examples.values()) * GRID_HW*GRID_HW), 3)}", end="\r")

            mean_loss = round(epoch_loss / n_batches, 5)
            mean_accuracy = round(sum(epoch_correct.values()) / (sum(n_examples.values()) * GRID_HW*GRID_HW), 3)
            accuracies_per_room = []
            for n in range(low_rooms, high_rooms + 1):
                if n in epoch_correct:
                    pct = epoch_correct[n] / (n_examples[n] * GRID_HW*GRID_HW)
                    accuracies_per_room.append((n, round(pct, 3)))
            print(f"epoch: {epoch}, mean_loss: {mean_loss} overall_mean_accuracy: {mean_accuracy}, per room: {accuracies_per_room}")

            # Persist Model checkpoint
            if epoch % 5 == 0:
                persister = VaePersister()
                with open(f"vae_softmax_finetune_{epoch}_{mean_loss}_{mean_accuracy}.pt", "wb") as file:
                    persister.persist_model(model=vae, io_handle=file)

        sys.exit(0)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)
