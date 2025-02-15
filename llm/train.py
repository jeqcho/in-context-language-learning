# %%
import os
import sys
import time
import yaml
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from llama import Llama
from data import generate_batch
from eval import ground_truth, unigram_batch, bigram_batch, evaluator_logits

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
    ):
        self.USE_DDP = use_ddp
        self.USE_MULTI_NODE = use_multi_node
        self.cfg = config
        self._unpack_dict(self.cfg)

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
            for key, value in d[section_key].items():
                setattr(self, key, value)

    def save_checkpoint(self, final=False):
        """Save model checkpoint."""
        self.model.eval()
        with t.no_grad():
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
            t.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        self.model.train()

    def load_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = t.load(checkpoint_path)
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

    def eval_step(self, to_log, num_states, evaluators=False, zipfian=False):
        """Performs a single evaluation step."""
        self.model.eval()
        with t.no_grad():
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
                zipfian=zipfian,
                eval=True,
            )

            # Move data to device with correct dtypes
            x_t = x_t.to(self.rank, dtype=t.int64)
            x_idx = x_idx.to(self.rank, dtype=t.int64)
            tm = tm.to(self.rank, dtype=t.float)

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
                    t.ones_like(preds_last, device=self.rank) / self.vocab_size
                ).to(self.rank, dtype=t.float)
                uniform_loss = self.eval_loss_fn(preds_last, uniform_probs)
                to_log[f"test/uniform_kl_div/{num_states}_states"] = (
                    uniform_loss.detach().cpu().numpy()
                )

                # Unigram Distribution Loss
                unigram_probs = unigram_batch(
                    x_idx[:, :-1], num_states, device=self.rank
                ).to(self.rank, dtype=t.float)
                unigram_logits = evaluator_logits(
                    self.vocab_size, unigram_probs, idx_to_token, self.rank
                ).to(self.rank, dtype=t.float)
                unigram_loss = self.eval_loss_fn(preds_last, unigram_logits)
                to_log[f"test/unigram_kl_div/{num_states}_states"] = (
                    unigram_loss.detach().cpu().numpy()
                )

                # Bigram Distribution Loss
                bigram_probs = bigram_batch(
                    x_idx[:, :-1], num_states, device=self.rank
                ).to(self.rank, dtype=t.float)
                bigram_logits = evaluator_logits(
                    self.vocab_size, bigram_probs, idx_to_token, self.rank
                ).to(self.rank, dtype=t.float)
                bigram_loss = self.eval_loss_fn(preds_last, bigram_logits)
                to_log[f"test/bigram_kl_div/{num_states}_states"] = (
                    bigram_loss.detach().cpu().numpy()
                )

        self.model.train()
        return eval_loss, to_log

    def train_step(self):
        """Performs a single training step."""
        start_time = time.time()
        self.model.train()

        to_log = {}

        # Generate batch and move to device
        x_t = generate_batch(
            self.train_batch_size,
            self.train_lang_size,
            self.seq_len,
            self.num_states,
            self.random_symbols,
            self.low_symbols,
            self.high_symbols,
            self.random_selection,
            self.low_idx,
            self.high_idx,
            self.doubly_stochastic,
            zipfian=self.zipfian,
            eval=False,
        ).to(self.rank, dtype=t.int64)

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

        for _ in range(self.steps + 1):
            to_log = self.train_step()  # Perform training step

            # Ensure only rank 0 performs evaluation
            if is_main_process and self.iters % self.eval_interval == 0:
                for num_states in eval_states:
                    eval_loss, to_log = self.eval_step(
                        to_log, num_states, evaluators=True, zipfian=False
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
            device_id = global_rank % t.cuda.device_count()

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
    optim = t.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=cfg["train"]["weight_decay"],
    )
    train_loss_fn = nn.CrossEntropyLoss()
    eval_loss_fn = nn.KLDivLoss(reduction="batchmean")

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
    )

    # Uncomment to load from checkpoint
    # trainer.load_from_checkpoint("/path/to/checkpoint")

    train_loss, eval_loss = trainer.fit()

    if USE_DDP:
        dist.destroy_process_group()  # Cleanup distributed processes


if __name__ == "__main__":
    main(sys.argv[1])
