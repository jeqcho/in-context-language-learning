# %%
import os
import sys
import time
import numpy as np
from pomegranate.hmm.dense_hmm import DenseHMM
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from torch.utils.data import Dataset, DataLoader

from hmm.HMMArgs import HMMArgs
from hmm.utils import HMMWrapper, load_model
from hmm.DataGenerator import DataGenerator

from llama import Llama
from data import generate_batch
from eval import ground_truth, unigram_batch, bigram_batch, evaluator_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class HMMDataset(Dataset):
    def __init__(self, input_ids):
        "Initialization"
        self.input_ids = input_ids

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.input_ids)

    def __getitem__(self, idx):
        "Generates one sample of data"
        # Select sample
        return self.input_ids[idx]


class MemoryMappedHMMDataset(Dataset):
    def __init__(self, data_filename, chunk_size=None):
        """Initialize dataset with memory mapping for efficient handling of large datasets.
        
        Args:
            data_filename: Path to the .npy file containing the dataset
            chunk_size: Optional size for data chunking. If None, determines a reasonable 
                        chunk size based on available memory.
        """
        # Use memory mapping to access the dataset without loading it entirely
        self.data = np.load(data_filename, mmap_mode='r')
        self.data_shape = self.data.shape
        
        # Set up chunking if requested
        self.chunk_size = chunk_size
        if self.chunk_size is None:
            # Determine a reasonable chunk size (default to 100,000 samples)
            self.chunk_size = 100000
        
        # Cache to store recently accessed chunks
        self.cache = {}
        self.max_cache_chunks = 5  # Maximum number of chunks to keep in memory
    
    def __len__(self):
        "Returns total number of samples"
        return self.data_shape[0]
    
    def __getitem__(self, idx):
        "Get sample by index using chunked access"
        # Calculate which chunk this index belongs to
        chunk_idx = idx // self.chunk_size
        
        # If chunk is not in cache, load it
        if chunk_idx not in self.cache:
            # If cache is full, remove least recently used chunk
            if len(self.cache) >= self.max_cache_chunks:
                oldest_chunk = next(iter(self.cache))
                del self.cache[oldest_chunk]
            
            # Load the chunk into cache
            start = chunk_idx * self.chunk_size
            end = min((chunk_idx + 1) * self.chunk_size, len(self))
            self.cache[chunk_idx] = torch.from_numpy(self.data[start:end].copy())
        
        # Get index within the chunk
        local_idx = idx % self.chunk_size
        return self.cache[chunk_idx][local_idx]


class Trainer:
    def __init__(
        self,
        model,
        optim,
        config,
        save_path,
        train_loss_fn,
        eval_loss_fn,
        rank,
        use_ddp,
        use_multi_node,
        data_loader,
    ):
        self.USE_DDP = use_ddp
        self.USE_MULTI_NODE = use_multi_node
        self.cfg = config
        self._unpack_dict(self.cfg)
        self.data_loader = data_loader

        self.rank = rank
        self.global_rank = int(os.environ.get("RANK", rank)) if use_multi_node else rank

        print(f"Running on global rank {self.global_rank}, local rank {self.rank}")

        self.model = model.to(device=self.rank)
        if use_ddp:
            self.model = DDP(self.model, device_ids=[self.rank])

        self.opt = optim
        self.save_path = save_path
        self.train_loss_fn = train_loss_fn
        self.eval_loss_fn = eval_loss_fn

        self.vocab_size = self.high_idx - self.low_idx
        self.iters = 0

        self.warmup_steps = int(self.warmup_ratio * self.steps)
        self.linear_scheduler_steps = int((1 - self.warmup_ratio) * self.steps)

        self.scheduler = LinearLR(
            self.opt,
            start_factor=self.linear_start_factor,
            end_factor=self.linear_end_factor,
            total_iters=self.steps,
        )

        print(f"Initial learning rate: {self.scheduler.get_last_lr()[0]}")

    def _unpack_dict(self, d):
        """Unpack config dictionary into attributes."""
        # d can be either a dict or a Run. Run have keys() but not values()
        for section_key in d.keys():
            # TODO overhaul the cfg loading
            if section_key == "hmm":
                continue
            for key, value in d[section_key].items():
                setattr(self, key, value)

    def save_checkpoint(self, final=False):
        """Save model checkpoint."""
        self.model.eval()
        with torch.no_grad():
            checkpoint_path = os.path.join(
                self.save_path, "final" if final else f"step_{self.iters}"
            )
            checkpoint_data = {
                "model": (
                    self.model.module.state_dict()
                    if self.USE_DDP
                    else self.model.state_dict()
                ),
                "opt": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "num_train": self.iters,
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        self.model.train()

    def load_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        model_state = checkpoint["model"]
        (
            self.model.module.load_state_dict(model_state)
            if self.USE_DDP
            else self.model.load_state_dict(model_state)
        )
        self.iters = checkpoint["num_train"]
        print(f"Loaded checkpoint from {checkpoint_path}, at step {self.iters}")

    def save_pretrained(self):
        """Save model in Hugging Face format."""
        pretrained_path = os.path.join(self.save_path, "final_hf")
        (
            self.model.module.save_pretrained(pretrained_path)
            if self.USE_DDP
            else self.model.save_pretrained(pretrained_path)
        )
        print(f"Saved Hugging Face model at {pretrained_path}")

    def eval_step(
        self,
        to_log,
        num_states,
        x_t,
        x_idx,
        tm,
        idx_to_token,
        evaluators=False,
        zipfian=False,
    ):
        """Performs a single evaluation step."""
        self.model.eval()
        with torch.no_grad():
            # Model forward pass
            preds = self.model(x_t)[:, :-1, :]  # Predict next tokens
            preds_log = F.log_softmax(preds, dim=-1)  # Convert to log probabilities

            # Extract last timestep predictions
            preds_last = preds_log[:, -1, :]
            truth = ground_truth(self.vocab_size, x_idx, tm, idx_to_token, self.rank)
            truth_last = truth[:, -1, :]

            # Compute KL divergence loss
            eval_loss = self.eval_loss_fn(preds_last, truth_last)
            zipf_suffix = "_zipf" if zipfian else ""
            to_log[f"test/kl_loss/{num_states}_states{zipf_suffix}"] = (
                eval_loss.detach().cpu().numpy()
            )

            if evaluators:
                # Uniform Distribution Loss
                uniform_probs = (
                    torch.ones_like(preds_last, device=self.rank) / self.vocab_size
                ).to(self.rank, dtype=torch.float)
                uniform_loss = self.eval_loss_fn(preds_last, uniform_probs)
                to_log[f"test/uniform_kl_div/{num_states}_states"] = (
                    uniform_loss.detach().cpu().numpy()
                )

                # Unigram Distribution Loss
                unigram_probs = unigram_batch(
                    x_idx[:, :-1], num_states, device=self.rank
                ).to(self.rank, dtype=torch.float)
                unigram_logits = evaluator_logits(
                    self.vocab_size, unigram_probs, idx_to_token, self.rank
                ).to(self.rank, dtype=torch.float)
                unigram_loss = self.eval_loss_fn(preds_last, unigram_logits)
                to_log[f"test/unigram_kl_div/{num_states}_states"] = (
                    unigram_loss.detach().cpu().numpy()
                )

                # Bigram Distribution Loss
                bigram_probs = bigram_batch(
                    x_idx[:, :-1], num_states, device=self.rank
                ).to(self.rank, dtype=torch.float)
                bigram_logits = evaluator_logits(
                    self.vocab_size, bigram_probs, idx_to_token, self.rank
                ).to(self.rank, dtype=torch.float)
                bigram_loss = self.eval_loss_fn(preds_last, bigram_logits)
                to_log[f"test/bigram_kl_div/{num_states}_states"] = (
                    bigram_loss.detach().cpu().numpy()
                )

        self.model.train()
        return eval_loss, to_log

    def train_step(self, x_t):
        """Performs a single training step."""
        start_time = time.time()
        self.model.train()

        to_log = {}

        # Forward pass
        preds = self.model(x_t)[:, :-1, :]  # Predict next tokens
        target = x_t[:, 1:]  # Shift targets for next-token prediction

        # Compute loss
        loss = self.train_loss_fn(preds.transpose(-1, -2), target)

        # Store loss as NumPy array
        self.train_loss.append(loss.detach().cpu().numpy())

        # Log single loss value
        to_log["train_ce_loss"] = loss.detach().cpu().numpy()

        # Backpropagation
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler.step()

        # Performance metrics
        elapsed_time = time.time() - start_time
        iters_per_sec = 1 / elapsed_time if elapsed_time > 0 else float("inf")
        tokens_per_sec = iters_per_sec * self.train_batch_size * self.seq_len

        # Store throughput metrics
        to_log.update(
            {
                "throughput/iterations_per_second/device": iters_per_sec,
                "throughput/tokens_per_second/device": tokens_per_sec,
                "train/lr": self.scheduler.get_last_lr()[0],
            }
        )

        self.iters += 1
        return to_log

    def fit(self):
        """Train the model, evaluate periodically, and save checkpoints."""

        is_main_process = (not self.USE_DDP) or (self.USE_DDP and self.global_rank == 0)

        if is_main_process:
            print("Starting training")

        self.model.train()
        self.train_loss = []
        self.eval_loss = []

        # Determine the evaluation range dynamically
        eval_states = list(range(3, 81, 5))  # Dynamically generated [3, 5, 10, ..., 80]

        num_epoch = 1
        for __ in range(num_epoch):
            for batch in self.data_loader:
                # Generate batch and move to device
                batch = batch.to(self.rank, dtype=torch.int64)

                to_log = self.train_step(batch)  # Perform training step

                # Ensure only rank 0 performs evaluation
                if is_main_process and self.iters % self.eval_interval == 0:
                    for num_states in eval_states:
                        with torch.no_grad():
                            # Generate batch
                            x_t, x_idx, tm, idx_to_token = generate_batch(
                                batch_size=self.eval_batch_size,
                                lang_size=self.eval_lang_size,
                                length=self.seq_len,
                                num_symbols=num_states,
                                random_symbols=False,
                                low_symbols=None,
                                high_symbols=None,
                                random_selection=self.random_selection,
                                low_idx=self.low_idx,
                                high_idx=self.high_idx,
                                doubly_stochastic=self.doubly_stochastic,
                                zipfian=False,
                                eval=True,
                            )

                            # Move data to device with correct dtypes
                            x_t = x_t.to(self.rank, dtype=torch.int64)
                            x_idx = x_idx.to(self.rank, dtype=torch.int64)
                            tm = tm.to(self.rank, dtype=torch.float)

                        eval_loss, to_log = self.eval_step(
                            to_log,
                            num_states,
                            x_t,
                            x_idx,
                            tm,
                            idx_to_token,
                            evaluators=True,
                            zipfian=False,
                        )
                        self.eval_loss.append(eval_loss.detach().cpu().numpy())

                # Checkpointing at intervals
                if is_main_process and self.iters % self.save_interval == 0:
                    self.save_checkpoint()

                # Logging metrics
                if is_main_process:
                    if self.iters % self.print_interval == 0:
                        print(f"Step {self.iters} completed")
                    if wandb.run is not None:
                        wandb.log(to_log)

        # Final checkpoint saving
        if is_main_process:
            self.save_checkpoint(final=True)
            self.save_pretrained()

        # Clean up W&B logging
        if wandb.run is not None:
            wandb.finish()

        return self.train_loss, self.eval_loss


def initialize_wandb(cfg):
    """Initialize Weights & Biases logging."""
    wandb_cfg = cfg.copy()
    wandb_cfg.pop("wandb")
    return wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        dir=cfg["wandb"]["wandb_dir"],
        config=wandb_cfg,
    )


def configure_dir(run_id, run_name, cfg):
    """Configure directory for saving model."""
    save_path = os.path.join(cfg["model"]["save_path"], f"{run_name}-{run_id}")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), "w") as file:
        yaml.dump(cfg, file)
    return save_path


def main(cfg_path):
    """Main execution function with DDP and multi-node support."""

    USE_DDP = True
    USE_MULTI_NODE = False  # Set to True if using multiple nodes

    # Default single-GPU setup
    global_rank = 0
    device_id = device

    if USE_DDP:
        dist.init_process_group("nccl")
        global_rank = dist.get_rank()

        if USE_MULTI_NODE:
            print(f"Running DDP on global rank {global_rank}.")
            device_id = int(os.environ["LOCAL_RANK"])
        else:
            print(f"Running DDP on rank {global_rank}.")
            device_id = global_rank % torch.cuda.device_count()

    # Load config
    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)

    # Only rank 0 handles logging and saving
    if (USE_DDP and global_rank == 0) or not USE_DDP:
        run = initialize_wandb(cfg)
        cfg = run.config
        save_path = configure_dir(run.id, run.name, cfg)
    else:
        save_path = None  # Other ranks don’t log or save

    print(f"Config: {cfg}")

    # Initialize model and optimizer
    model = Llama(cfg).to(device=device_id)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=cfg["train"]["weight_decay"],
    )
    train_loss_fn = nn.CrossEntropyLoss()
    eval_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # Init dataloader
    # TODO put filename params into the yaml

    hmm_args = HMMArgs(
        num_emissions=cfg["hmm"]["num_emissions"],
        num_states=cfg["hmm"]["num_states"],
        seq_length=cfg["hmm"]["seq_length"],
        batch_size=cfg["hmm"]["batch_size"],
        unique=cfg["hmm"]["unique"],
        update_freq=cfg["hmm"]["update_freq"],
        num_epoch=100,  # doesn't matter
    )

    hmm_model = load_model(hmm_args, epoch_on_filename=cfg["hmm"]["load_model_with_epoch"])

    hmm_wrapper = HMMWrapper(
        hmm_model, hmm_args
    )  # file naming uses hmm_args only and not the model

    data_generator = DataGenerator(
        num_seq=cfg["hmm"]["num_seq"],
        gen_seq_len=cfg["hmm"]["gen_seq_len"],
        permutate_emissions=cfg["hmm"]["permutate_emissions"],
        hmm_wrapper=hmm_wrapper,
        epoch_on_filename=cfg["hmm"]["load_model_with_epoch"],
        suffix=cfg["hmm"]["suffix"],
    )
    
    # Use memory-mapped dataset for efficient handling of large data
    print(f"Loading dataset from {data_generator.data_filename} using memory mapping")
    # Calculate appropriate chunk size based on batch size and sequence length
    chunk_size = cfg["train"]["train_batch_size"] * 500  # Larger than batch size for efficiency
    
    # Create dataset with memory mapping
    dataset = MemoryMappedHMMDataset(
        data_filename=data_generator.data_filename,
        chunk_size=chunk_size
    )
    
    # Configure optimized DataLoader with multiple workers
    num_workers = min(8, os.cpu_count() or 1)  # Use multiple workers but not too many
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg["train"]["train_batch_size"],
        shuffle=True,
        num_workers=num_workers,  # Use multiple CPU workers for data loading
        pin_memory=True,  # Pin memory for faster GPU transfer
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    del hmm_model

    # Trainer instance
    trainer = Trainer(
        model,
        optim,
        cfg,
        save_path,
        train_loss_fn,
        eval_loss_fn,
        device_id,
        USE_DDP,
        USE_MULTI_NODE,
        data_loader,
    )

    # Uncomment to load from checkpoint
    # trainer.load_from_checkpoint("/path/to/checkpoint")

    train_loss, eval_loss = trainer.fit()

    if USE_DDP:
        dist.destroy_process_group()  # Cleanup distributed processes


if __name__ == "__main__":
    main(sys.argv[1])

# %%
