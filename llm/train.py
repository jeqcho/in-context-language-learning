# %%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
from tqdm import tqdm
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# from gpt2 import GPT2
# from rope import RoFormer
from llama import Llama

from data import generate_batch
from eval import ground_truth, unigram_batch, bigram_batch, evaluator_logits
# from bigram import unigram_batch, bigram_batch
import yaml
import sys

import os
import time
import wandb

import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

class Trainer:

    def __init__(self, model, optim,  config, save_path, train_loss_fn, eval_loss_fn, rank, USE_DDP, USE_MULTI_NODE):

        self.USE_DDP = USE_DDP
        self.USE_MULTI_NODE = USE_MULTI_NODE

        self.cfg = config
        self._unpack_dict(self.cfg)

        self.rank = rank
        self.global_rank = rank
        
        if self.USE_MULTI_NODE:
            self.global_rank = int(os.environ['RANK'])
            print("Running on global rank", self.global_rank, 'and local rank', self.rank)

        self.model = model
        self.model.to(device=self.rank)

        # REMOVE LATER
        # FREEZE EVERYTHING EXCEPT LAST MLP LAYER
        # print('freezing everything except last mlp layer')
        # for name, param in model.named_parameters():
        #     if 'mlp' not in name:
        #         param.requires_grad = False
        #     if 'lin2' in name:
        #         param.requires_grad = True

        # print('listing trainable params')
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print('training', name, param.shape)

        if self.USE_DDP:
            self.model = DDP(self.model, device_ids=[self.rank])

        self.opt = optim
        self.save_path = save_path

        self.warmup_steps = int(self.warmup_ratio * self.steps)
        self.linear_scheduler_steps = int((1 - self.warmup_ratio) * self.steps)

        # self.warmup_scheduler = LinearLR(self.opt, start_factor=1.0, end_factor=1.0, total_iters=self.warmup_steps, last_epoch=-1)
        # self.train_scheduler = LinearLR(self.opt, start_factor=self.linear_start_factor, end_factor = self.linear_end_factor, total_iters=self.linear_scheduler_steps, last_epoch=-1)
        # self.scheduler = SequentialLR(self.opt, schedulers=[self.warmup_scheduler, self.train_scheduler], milestones=[self.warmup_steps])

        print('pre scheduler lr', self.lr)
        self.scheduler = LinearLR(self.opt, start_factor=self.linear_start_factor, end_factor=self.linear_end_factor, total_iters=self.steps, last_epoch=-1)
        print('first learning rate', self.scheduler.get_last_lr()[0])

        self.train_loss_fn = train_loss_fn
        self.eval_loss_fn = eval_loss_fn

        self.vocab_size = self.high_idx - self.low_idx
        self.iters = 0

    def _unpack_dict(self, d):
        for section_key in d.keys():
            for k, v in d[section_key].items():
                setattr(self, k, v)
    
    def save_checkpoint(self, final=False):
        self.model.eval()
        with t.no_grad():
            checkpoint_path = self.save_path + 'step_' + str(self.iters)
            if final:
                checkpoint_path = self.save_path + 'final'

            if self.USE_DDP:
                t.save({"model": self.model.module.state_dict(),
                        "opt": self.opt.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "num_train": self.iters}, checkpoint_path)
            else:
                t.save({"model": self.model.state_dict(),
                        "opt": self.opt.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "num_train": self.iters}, checkpoint_path)

            if final:
                print('Saved final checkpoint (torch) at %s'%checkpoint_path)
            else:
                print("Checkpoint for step " + str(self.iters) + " saved in " + checkpoint_path)
        self.model.train()

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = t.load(checkpoint_path)

        if self.USE_DDP:
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        # self.opt.load_state_dict(checkpoint["opt"])
        # self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.iters = checkpoint["num_train"]
        print("Loaded checkpoint from %s"%checkpoint_path)
        print("Currently at step %d"%self.iters)

    def save_pretrained(self):
        pretrained_path = self.save_path + 'final_hf'

        if self.USE_DDP:
            self.model.module.save_pretrained(pretrained_path)
        else:
            self.model.save_pretrained(pretrained_path)

        print('Saved final checkpoint (huggingface) at %s'%pretrained_path)

    def eval_step(self, to_log, num_states, evaluators=False, zipfian=False):
        self.model.eval()
        with t.no_grad():

            x_t, x_idx, tm, idx_to_token = generate_batch(batch_size=self.eval_batch_size, lang_size=self.eval_lang_size, length=self.seq_len, num_symbols=num_states, random_symbols=False, low_symbols=None, high_symbols=None, random_selection=self.random_selection, low_idx=self.low_idx, high_idx=self.high_idx, doubly_stochastic=self.doubly_stochastic, zipfian=zipfian, eval=True)

            x_t = x_t.to(device=self.rank, dtype=int)
            x_idx = x_idx.to(device=self.rank, dtype=int)
            tm = tm.to(device=self.rank, dtype=float)

            test_starting_at = 0
            preds = self.model(x_t)[:, test_starting_at:-1, :]
            preds_log = F.log_softmax(preds, dim=-1)
            truth = ground_truth(self.vocab_size, x_idx, tm, idx_to_token, self.rank)

            preds_last = preds_log[:, -1, :]
            truth_last = truth[:, -1, :]

            zipf_text = '' if not zipfian else '_zipf'

            eval_loss = self.eval_loss_fn(preds_last, truth_last)
            to_log["test/kl_loss/" + str(num_states) + '_states' + zipf_text] = eval_loss.detach().cpu().numpy()

            if evaluators:

                # uniform loss
                uniform_probs = (t.ones_like(preds_last) / self.vocab_size).to(device=self.rank, dtype=float)
                uniform_kl_div = self.eval_loss_fn(preds_last, uniform_probs)
                to_log["test/uniform_kl_div/" + str(num_states) + '_states'] = uniform_kl_div.detach().cpu().numpy()
            
                # unigram loss
                unigram_probs = unigram_batch(x_idx[:, :-1], num_states, device=self.rank).to(device=self.rank, dtype=float)
                unigram_logits = evaluator_logits(self.vocab_size, unigram_probs, idx_to_token, self.rank).to(device=self.rank, dtype=float)
                unigram_kl_div = self.eval_loss_fn(preds_last, unigram_logits)
                to_log["test/unigram_kl_div/" + str(num_states) + '_states'] = unigram_kl_div.detach().cpu().numpy()

                # bigram loss
                bigram_probs = bigram_batch(x_idx[:, :-1], num_states, device=self.rank).to(device=self.rank, dtype=float)
                bigram_logits = evaluator_logits(self.vocab_size, bigram_probs, idx_to_token, self.rank).to(device=self.rank, dtype=float)
                bigram_kl_div = self.eval_loss_fn(preds_last, bigram_logits)
                to_log["test/bigram_kl_div/" + str(num_states) + '_states'] = bigram_kl_div.detach().cpu().numpy()

            self.model.train()
            return eval_loss, to_log

    def train_step(self):
        start_time = time.time()
        self.model.train()
        
        to_log = {}

        x_t = generate_batch(self.train_batch_size, self.train_lang_size, self.seq_len, self.num_states, self.random_symbols, self.low_symbols, self.high_symbols, self.random_selection, self.low_idx, self.high_idx, self.doubly_stochastic, zipfian=self.zipfian, eval=False).to(device=self.rank, dtype=int)

        # x_t, x_idx, tm, idx_to_token = generate_batch(self.train_batch_size, self.train_lang_size, self.seq_len, self.num_states, self.random_symbols, self.low_symbols, self.high_symbols, self.random_selection, self.low_idx, self.high_idx, self.doubly_stochastic, zipfian=self.zipfian, eval=True)

        # x_t = x_t.to(device=self.rank, dtype=int)
        # x_idx = x_idx.to(device=self.rank, dtype=int)
        # tm = tm.to(device=self.rank, dtype=float)
        
        train_starting_at = 0
        preds = self.model(x_t)[:, train_starting_at:-1, :]
        loss = self.train_loss_fn(preds.mT, x_t[:, train_starting_at+1:])

        # SETUP FOR KL DIV LOSS
        # bigram ground truth
        # preds_log = F.log_softmax(preds, dim=-1)
        # bigram_output = bigram_batch(x_t[:, :-1], self.high_symbols, device=self.rank).to(device=self.rank, dtype=float)
        # bigram_probs = evaluator_logits(self.vocab_size, bigram_output, idx_to_token, self.rank).to(device=self.rank, dtype=float)

        # loss = self.train_loss_fn(preds_log, bigram_probs)

        # loss = self.train_loss_fn(preds.mT, x_t[:, train_starting_at+1:])
        self.train_loss.append(loss.detach().cpu().numpy())

        to_log["train_ce_loss"] = loss.detach().cpu().numpy()
            
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        end_time = time.time()

        iters_per_sec = 1 / (end_time - start_time)
        tokens_per_sec = iters_per_sec * self.train_batch_size * self.seq_len

        to_log["throughput/iterations_per_second/device"] = iters_per_sec
        to_log["throughput/tokens_per_second/device"] = tokens_per_sec
        to_log["train/lr"] = self.scheduler.get_last_lr()[0]

        self.iters += 1

        self.scheduler.step()

        return to_log


    def fit(self):

        if (self.USE_DDP and self.global_rank == 0) or not self.USE_DDP:
            print("Starting training")

        self.model.train()

        self.train_loss = []
        self.eval_loss = []

        # calculate max num_states to see bigrams 3 times
        max_num_states = min(self.vocab_size, int((self.seq_len // 3) ** (1/2)))
        print('max num states', max_num_states)

        for _ in range(self.steps+1):

            to_log = self.train_step()

            if (self.USE_DDP and self.global_rank == 0) or not self.USE_DDP:

                if self.iters % self.eval_interval == 0:

                    eval_loss, to_log = self.eval_step(to_log, 3, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 5, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 8, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    # eval_loss, to_log = self.eval_step(to_log, self.num_states, evaluators=True, zipfian=False)
                    # self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 15, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 20, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 25, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 30, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 35, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 40, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 45, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 50, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 55, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 60, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 65, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 70, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 75, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    eval_loss, to_log = self.eval_step(to_log, 80, evaluators=True, zipfian=False)
                    self.eval_loss.append(eval_loss.detach().cpu().numpy())

                    # eval_loss, to_log = self.eval_step(to_log, max_num_states, evaluators=False, zipfian=False)
                    # self.eval_loss.append(eval_loss.detach().cpu().numpy())

                if self.iters % self.print_interval == 0:
                    print(f"Step {self.iters} completed")
                
                if self.iters % self.save_interval == 0:
                    self.save_checkpoint()

                if wandb.run is not None:
                    wandb.log(to_log)

        if (self.USE_DDP and self.global_rank == 0) or not self.USE_DDP:
            self.save_checkpoint(final=True)
            self.save_pretrained()
        
        if wandb.run is not None:
            wandb.finish()

        return self.train_loss, self.eval_loss
    
def initialize_wandb(cfg):
    wandb_cfg = cfg.copy()
    wandb_cfg.pop('wandb')
    run = wandb.init(
        project=cfg['wandb']['project'],
        entity=cfg['wandb']['entity'],
        dir=cfg['wandb']['wandb_dir'],
        config=wandb_cfg)
    return run

def configure_dir(run_id, run_name, cfg):
    save_path = cfg['model']['save_path']
    save_path += run_name + '-' + run_id + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory {save_path} created")
        with open(save_path + 'config.yaml', 'w') as file:
            yaml.dump(cfg, file)
            print(f"Config file saved in {save_path + 'config.yaml'}")
    return save_path

# %%

def main(cfg_path):

    USE_DDP = True
    USE_MULTI_NODE = False

    # # SINGLE GPU
    global_rank = 0
    device_id = device

    if USE_DDP:
        dist.init_process_group("nccl")
        global_rank = dist.get_rank()
    
        if USE_MULTI_NODE:
            print(f"Running DDP on global rank {global_rank}.")
            device_id = int(os.environ['LOCAL_RANK'])
        else:
            print(f"Running DDP on rank {global_rank}.")
            device_id = global_rank % t.cuda.device_count()

    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    if (USE_DDP and global_rank == 0) or not USE_DDP:
        run = initialize_wandb(cfg)
        cfg = run.config
        save_path = configure_dir(run.id, run.name, cfg)
    else:
        save_path = None

    print('cfg', cfg)

    model = Llama(cfg).to(device=device_id)

    optim = t.optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=cfg['train']['weight_decay'])
    train_loss_fn = nn.CrossEntropyLoss()
    # train_loss_fn = nn.KLDivLoss(reduction='batchmean')
    eval_loss_fn = nn.KLDivLoss(reduction='batchmean')

    trainer = Trainer(model, optim, cfg, save_path, train_loss_fn, eval_loss_fn, device_id, USE_DDP, USE_MULTI_NODE)

    # trainer.load_from_checkpoint('/n/holyscratch01/sham_lab/summer_2024/checkpoints/upbeat-brook-28-rejwbgtm/final')

    train_loss, eval_loss = trainer.fit()

    if USE_DDP:
        dist.destroy_process_group()

# %%

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    # cfg_path = 'wandb_config.yaml'
    main(cfg_path)
    # %%