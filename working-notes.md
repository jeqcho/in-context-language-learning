# Jan 20

This continues the work from `29-Dec-4/README.md`.

To recap, we have the TinyStories datsets cleaned and tokenized on a word level. We included datasets for sentences using the top 100, 200, 300, 400, 500 words. Here is the one for top 100.
- `/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-100-train.txt`
- `/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-100-test.txt`

The file looks like this

```
18 11 4 3 27 54 22 19 40 5 1 62 0
7 36 24 31 0
18 11 4 3 54 22 19 40 5 1 62 5 29 0
18 11 4 3 27 54 22 19 40 5 1 62 13 14 46 0
8 17 3 55 22 87 0
7 33 3 16 50 12 3 61 0
95 4 7 40 82 5 45 68 96 0
```

The final step is to train the HMMs. This folder will train the HMMs using GPU via pomegranate.

We will fit the HMMs with the following parameters:
- Emissions (E): 100, 200, 300, 400
- Hidden states (H): 100, 200, 300, 400
- Sequence lengths (L): 100, 200, 300, 400

This work is done at `train_hmm.py`.

Testing on 1000 to 4000 rows, it seems like the model can train under 3 minutes. I will test this out.

Kernel crashed when I ran the run for 65k rows. Could be memory issue.

Using GPU, extrapolation suggests 18.45 seconds.

I ran into GPU issues. We can try to subset it so that it fits on a GPU, or a sbatch for 4 H100 GPUs which gives 320GB, which can fit our 243 GB.

# Jan 21

Let's try to sbatch it on 2 GPUs. I will keep testing until it breaks, and then try to submit that to 2 GPUs (which theoretically should run without errors).

28.98 GiB is free.
24.46 GiB is free.
22.91 GiB is free

65930 rows gives 243.15 GiB
30000 rows gives 110.64 GiB
10000 rows gives 36.88 GiB

This gives roughly 270 rows per GB

Let's try to fit 35 GB. This wouldn't pass 1 GPU, but should fit in 2 GPUs.
270*35 = 9450 rows

20 GB is 5400 rows

First try on 1 GPU
Tried to allocate 34.85 GiB. GPU 0 has a total capacity of 39.38 GiB of which 21.45 GiB is free.

Second try on 1 GPU.
Tried to allocate 34.85 GiB. GPU 0 has a total capacity of 39.38 GiB of which 2.54 GiB is free.

Since we are testing H100s. We will test 1 GPU first with 9450 rows.
Now we will sbatch using `test_1_gpu.sh`. This should pass because 35GB should fit.

Surprisingly, it didn't pass.

```
Tried to allocate 34.85 GiB. GPU 0 has a total capacity of 79.10 GiB of which 6.90 GiB is free. Including non-PyTorch memory, this process has 72.18 GiB memory in use. Of the allocated memory 36.69 GiB is allocated by PyTorch, and 34.84 GiB is reserved by PyTorch but unallocated.
```

I suspect it has to do with the tensor slicing. I now modify it so I move the tensor to gpu after slicing.

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 34.85 GiB. GPU 0 has a total capacity of 79.10 GiB of which 6.93 GiB is free. Including non-PyTorch memory, this process has 72.15 GiB memory in use. Of the allocated memory 36.65 GiB is allocated by PyTorch, and 34.85 GiB is reserved by PyTorch but unallocated.
```

So moving to GPU after doesn't help.

I am thinking if it could be that they have to make a copy during training, so in practice we need twice. Let's see if it works if I don't move it to gpu.

Ok I need to specify the devices for both. Otherwise if not for both I get CPU which gives 2.7s instead of 0.5s.

Ok let's see what's the threshold.

3000 ok 36.1GB
4000 not ok 14.62 GiB reserved
6000 not ok

Let's try submittting 4k to h100. This works. Let's make it 6k. 6k works. Check 8k. Doesn't work.
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 29.51 GiB. GPU 0 has a total capacity of 79.22 GiB of which 18.02 GiB is free. Including non-PyTorch memory, this process has 61.19 GiB memory in use. Of the allocated memory 31.03 GiB is allocated by PyTorch, and 29.50 GiB is reserved by PyTorch but unallocated.
```
Let's try 2 GPUs and if that works out of the box. Unfortunately, it doesn't work out of the box.

Reading the docs again, I realized that the first dimension is batch size.

I will now write a train_loader.

# Jan 22

The run yesterday ran out of time once it got into the nan phase. I will put a `max_iter`, and also look at the pull request on github.

I pulled the pull request and install the local copy into pip. Let's test this out now.

The bug still exists.

Putting a `max_iter` won't solve the underlying problem. Let's check our initialization.

We can see if the problem still arises with CPU, or we can check if just using plain lists helps.

Submitted the CPU with sbatch. (doesn't help)

Just using plain list for the init doesn't help.

I asked o1 and it suggested using eps for the logs. I also added a code to warn loudly when its nan.

# Jan 23

Traced the error to `[WARNING] Infs found in emissions _check_inputs _base 3`. The code in question is `emissions = model._emission_matrix(X, priors=priors)` in `_base.py` under `_check_inputs`.

Solved the bug.

Next steps: add shuffling, look at this [link](https://pomegranate.readthedocs.io/en/latest/tutorials/C_Feature_Tutorial_3_Out_Of_Core_Learning.html) for batch training, and also sanity check predictions (from the output model).

# Jan 24

Use torch's `TensorDataset` and `DataLoader`. Writing evals at `sanity_check_hmm.py`.

There's a possible bug as the training loop freezes at 94% on epoch 2.

# Jan 25

Let's try to do 5 epochs using batched training. Works! Sent a sbatch for 10 epochs on a h100. In the meantime, let's sanity check.

I will write some helper functions in `utils.py` to help with tokenizing and detokenizing.

Note that from previous work, the tokenizer is stored at `/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/tokenizer.json`.

# Jan 26

Sbatch for 10 epochs work. Now focus on sanity checking.

It seems like the hidden probabilities are all the same. i.e. there has been no training.

Sum of probabilities of emissions for each word. Note that they don't sum to one and have a monotonically decreasing pattern.
```
tensor([8.3801e+00, 2.9094e+00, 1.0502e+00, 3.6931e+00, 5.7183e-01, 5.5644e-01,
        9.6844e-01, 1.0000e-06], device='cuda:0')
```

Let's check if using `fit` instead of `summaries` will work better.

It shows the exact thing.

Checking the output prediction for each epoch shows that it is indeed just outputting 0 as the emission.

Let's see if random init helps.

Great! Looks like the tokens are different at the very end of the context.

Tried training this on h100 but failed
```
hmm_args = HMMArgs(num_emissions=100, num_states=200, seq_length=300, batch_size=1024, num_epoch=5)
```

will use 512 now.

This throws error
```
hmm_args = HMMArgs(num_emissions=100, num_states=400, seq_length=300, batch_size=512, num_epoch=10)
```
but 256 should be fine. Still failed, trying 128.

Next steps. Check if random init the edges help too.

# Jan 27

754644 took 36 minutes while 755199 took 60 minutes.

For 755199
```
HMMArgs(num_emissions=100, num_states=400, seq_length=600, batch_size=64, num_epoch=10)
```

Let's try to debug the pbar first.

Should be ok. Added epoch to the file name.

Submitted two training runs.

I think `self.k` in the `pomegranate` library refers to the number of emissions. Hmm, I take that back. There's this line.

```
@property
def k(self):
        return len(self.distributions) if self.distributions is not None else 0

@property
def n_distributions(self):
        return len(self.distributions) if self.distributions is not None else 0
```

I think it will be more clear if we have diff dimensions. I will submit a quick run.

The periods are always correct.

But if we look at the definition of `self.k`, it seems like the returned shape to be `"batch seq_len n_hidden"` instead of `"batch seq_len n_emissions"`.

For
```
HMMArgs(num_emissions=100, num_states=200, seq_length=300, batch_size=256, num_epoch=1)
```
the commas and periods are accurately predicted. Note the single epoch.

Sbatch runs for
882840
```
HMMArgs(num_emissions=100, num_states=200, seq_length=600, batch_size=256, num_epoch=10)
```

882789
```
HMMArgs(num_emissions=100, num_states=200, seq_length=300, batch_size=256, num_epoch=10)
```

883266
```
HMMArgs(num_emissions=100, num_states=400, seq_length=300, batch_size=128, num_epoch=10)
```

It turns out that the returned shape is indeed `"batch seq_len n_hidden"`.

Then, what's the difference between `_emission_matrix` and `predict_log_proba`? I think the main difference is that `predict_log_proba` takes into account the entire sequence, while `_emission_matrix` doesn't.

We can still grab the "real" emission matrix by doing `model.distributions[i].prob`.

Why is `model.distributions[state].probs[0]` having size `(1, 100)`? Shouldn't it be `(100,1)`, since we have this from `Categorical`
```
probs: list, numpy.ndarray, torch.tensor or None, shape=(k, d), optional
        Probabilities for each key for each feature, where k is the largest
        number of keys across all features. Default is None
```

todo-done: answer the above.

Also, check out if the HMM is learning the uniform, unigram, bigram strat.

# Jan 28

Checking the shapes, and see if our init is correct.

This is fine, since
```
probs: list, numpy.ndarray, torch.tensor or None, shape=(k, d), optional
Probabilities for each key for each feature, where k is the largest number of keys across all features. Default is None

n_categories: list, numpy.ndarray, torch.tensor or None, optional
The number of categories for each feature in the data. Only needs to be provided when the parameters will be learned directly from data and you want to make sure that right number of keys are included in each dimension. Default is None.
```

Since each `Categorical` represents a feature, the number of feature is 1, so the shape is indeed `(1, 100)`.

We can create a 2D heatmap to visualize how n_hidden and seq_length helps with test cross-entropy of the last token.

Sanity check our understanding of `_emission_matrix` as prior. So it should correspond to a hidden state that has the highest emission probability for this emission.

Somehow, the emission probability for the first (index 0) emission is zero across all hidden states (which is the period .). Let me check initialization. Init looks fine. Let's check another model. Other model looks fine.

Idea: see how the "name hidden state" evolve over num_hidden, seq_len and epochs.

Training a small model, the zero probability occured immediately after the first epoch. Let's see if this happens with `fit`.

Also check if using `fit` does different things.

# Jan 29

Two tasks
1. Check the index 0 emission having zero probability bug
2. Check if `fit` gives similar performance. (I think we can discard this for now).

On `fit`, the run at 1054056 uses `fit` with params `HMMArgs(num_emissions=100, num_states=200, seq_length=300, batch_size=256, num_epoch=10)`, but it didn't manage to train even one epoch within 2 hours. Final estimation is around 5 hours. I think `fit` and `from_summaries` do the same thing, and `from_summaries` is faster.

Let's continue to solve the index 0 zero probability emission bug. Let's try with `fit`.

`fit` is taking a long time to converge for a single batch. I am wondering if it caused by the `EPS` I set, since back then the steps move pretty quickly. Let me test this out by setting `EPS` to 0. It got `nan` pretty quickly. I will just put a lower `EPS`. I set it to `EPS = 1e-14` since `sys.float_info.epsilon` is usually `2e-16`.

Would calling `summarize` and `from_summaries` both in one step work? Well, it's instant, as compared to `fit`. Is it because of `verbose`? Doesn't seem like it. Ran for a minute and it still couldn't get past it.

The one step `from_summaries` immediately threw an error for non-positive probability for first emission.

Let's check what happens in `summarize` and `from_summaries`.

Tracing...

The first element of `_xw_sum` is zero in `Categorical`.

Ok. It could be our fault. There's no 0 in the `batch`. Solved the problem. Now let's remove all previously trained models.

I don't have GCM or GitHub CLI, but somehow I can still push without entering my password. I need to look into this. TODO.

Took roughly 7 minutes to train `HMMArgs(num_emissions=100, num_states=100, seq_length=100, batch_size=1024, num_epoch=10)`.

Look's like the model is employing unigram strategy and is outputting the most common words like `.`, `once`, `upon`, `a`, `time`, etc.

I will train a bunch of models for our use later.

- `HMMArgs(num_emissions=100, num_states=200, seq_length=300, batch_size=256, num_epoch=10)` params ok, submitted
- `HMMArgs(num_emissions=100, num_states=200, seq_length=600, batch_size=256, num_epoch=10)` params ok, submitted
- `HMMArgs(num_emissions=100, num_states=400, seq_length=300, batch_size=128, num_epoch=10)` params ok, submitted
- `HMMArgs(num_emissions=100, num_states=200, seq_length=300, batch_size=256, num_epoch=20)` params ok, submitted
- `HMMArgs(num_emissions=100, num_states=200, seq_length=300, batch_size=256, num_epoch=40)` params ok, submitted

I should also keep track of the total time it takes to run each of them.

I also need to make it so that the command args are fed from the `.sh` so we don't have to wait for the job to run to get another one up running.

# Jan 30

I will first build the tool so that we can feed in command args.

I will now fix the print string for `HMMArgs`.

The last assert here couldn't pass
```
# get the probabilites for the final hidden state
hidden_state_prob = self.model.predict_proba(batch)
assert hidden_state_prob.shape == (b, s, h)
# check each row sums to one
print(hidden_state_prob)
assert torch.allclose(hidden_state_prob.sum(-1), torch.tensor(1.0))
```

This even returned 0.3345
```
hidden_state_prob[torch.where(hidden_state_prob.sum(-1) != 1.0)].max()
```

Turns out setting `atol=1e-5` passes.

The `DenseHMM.edges` are log probabilities, and exponentiating then summing them yields probs like 0.9888. Check this with `atol=5e-2`.

Note that these are the same but using two different batches
```
max diff from 1 for hidden_state_prob: 1.5497207641601562e-05
max diff from 1 for edges: 0.011353731155395508
max diff from 1 for next_state_prob: 2.384185791015625e-07
max diff from 1 for hidden_state_prob: 1.5497207641601562e-05
max diff from 1 for edges: 0.011353731155395508
max diff from 1 for next_state_prob: 2.384185791015625e-07
```

TODO check this if there's a bug.

# Jan 31

I will try to complete the `get_final_token_cross_entropy` function first. We can then see if it returns the same answers to different batches.

Great! We calculated the CE loss and it's different across batches.

# Feb 6

Back to working on this project. Will add wandb logging to track epoch effects and schedule more runs.

# Feb 7

Got the results from wandb. Seems like sequence length doesn't matter, but number of states matter. Fixing sequence length now and scheduling a sweep for number of states. Sbatched.

At the end of today, check back the big states small batches and see if we can speed them up with bigger batches.

NUM_STATES=(100 200 300 400 500 600)
BATCH_SIZES=(1024 256 128 128 64 64)
EST TIME BATCH=(30s, 3m, 13m, 17m, 27m, 32m)
EST TIME COMP=(25m, 2h30m, 10h50m, 14h10m, 22h30m, 26h40m)

We gave a budget of 4 hours.

I cancelled the last three runs and resubmitted with bigger batch sizes and time budget.
NUM_STATES=(100 200 300 400 500 600)
BATCH_SIZES=(1024 256 256 256 128 128)
EST TIME BATCH=(30s, 3m, 4.5m, 9m, 14m, 17m)
EST TIME COMP=(25m, 2h30m, 3h45m, 7h30m, 11h40m, 14h10m)

I also added batch size to the name of the models. Let's rerun everything at the end of today.

Now, let's try training with unique sentences instead.

All scheduled.

# Feb 8

Good plots! `tokens_seen` did not get logged. Try again with that. Also might be good to get the test loss from the unique dataset too.

Then we also have to chech why `tokens_seen` is not logged. Sbatched a quick run to see.

We also have to check why the runs failed. The H-200 case is at epoch 19 with:

`assert torch.allclose(transition_matrix.sum(-1), torch.tensor(1.0), atol=5e-2)`

same error for H-300 at epoch 15, H-400 at epoch 15, H-500 at epoch 16, H-600 at epoch 16, H-300-unique at epoch 21.

I moved two assert statements on distributions to after they are normalized.

Fix all bugs.

Now I will schedule a big run to scan across H, and also toggle unique.

I will run at H-100 for 200 epochs.

Now I will run for the uniques. I will need a new estimate for the durations.

NUM_STATES=(100 200 300 400 500 600)
BATCH_SIZES=(1024 256 256 256 128 128)
EST TIME BATCH=(30s, 3m, 4.5m, 9m, 14m, 17m)
EST TIME COMP=(25m, 2h30m, 3h45m, 7h30m, 11h40m, 14h10m)
EST TIME COMP=(25m, 150m, 225m, 450m, 700m, 850m)

New total sequences is 6899, old one is 74514, so it should be 10x faster, i.e. taking 10% of the time.

EST TIME COMP=(3m, 15m, 23m, 45m, 70m, 85m)
Actual time for the first is 4m+. I think 2h is good.

Submitted the rest as a sequential sbatch.

Now I will proceed to write the code for measuring KL divergence with other strategies like uniform, unigram, bigram.

Since each test is the same, we can just evaluate the logits once for each strategy and compare the HMM logits for each test round.

It seems like the HMM learns much more efficiently (by measuring tokens) if we train them on unique sequences instead of duplicated ones.

I will train a unique HMM with more epochs. 50m for 500 epochs. Sent.

Now, I will write the KL divergence stuff. Begin with uniform. Uniform done. For unigram, can check this [link](https://discuss.pytorch.org/t/batched-bincount/72819/2).

# Feb 9

Run results seem to suggest that training on the duplicated dataset is worth it.

Might be worth running a long run for H-600 unique to see if anything extraordinary happens. Submitted.

Might be worth checking if changing seq_len really doesn't matter. Will train a H-500 on L-100 to L-600.

Prev
NUM_STATES=(100 200 300 400 500 600)
BATCH_SIZES=(1024 256 256 256 128 128)

Result for H-500 E-100
SEQ_LEN=(100 200 300 400 500 600)
BATCH_SIZES=(256 128 64 64 32 32)

Not maximized for H-500 L-100
EMISSIONS=(100 200 300 400 500)
BATCH_SIZES=(256 256 256 256 256)

# Feb 10

It takes longer to train for larger emissions. I will set a max epoch size.

I am testing between training a HMM over an entire epoch before calling `from_summaries` vs calling `from_summaries` every step.
- time taken: 4m15s vs 4m05s
- test loss after 5 epochs: 4.57892 vs 4.04427
- unique test loss after 5 epochs: 4.58725 vs 4.30929

Let's do another experiment where we update every 2 steps.
- time taken: 4m02s
- test loss after 5 epochs: 4.04032
- unique test loss after 5 epochs: 4.30041

It could be possible that updating every step reaches a local minimum faster than updating every epoch reaching a global minimum.

Double check that no testing gives the same loss. Then try to optimize testing with no_grad (TODO).
- This seems to be different, which is worrying.
- Set a seed, and sbatched two runs.
- This looks diff.
- I will sbatch another pair and see if the diff is significant.

Also will sbatch a large run to scan over update_freq. Sent.

# Feb 11

Changing test frequency does change the exact loss. Let's try to see if this is a bug.

I don't think setting inference_mode will solve this bug, since we don't use gradients anyways. Let's see what `from_summaries` does. Is it affected by `predict_proba`?

Pretty much so. It calls `forward_backward` under the hood.

Idea: clear the things set by `forward_backward` after we eval. This is similar to how if we run `from_summaries` twice consecutively, there will be nan. Let's check `from_summaries`.

It seems that this is achieved with `reset_cache`.

Let me run a run that only does a test after 20 epochs, and compare that to one that tests every epoch but uses `reset_cache`. It is hoped that they will agree on epoch 20.

Note, if we use E-100 and B-1024, there's 72.7 batches per epoch.

The `reset_cache` doesn't look too promising, since its trajectory is similar to one without `reset_cache`. Hmm, I take that back. What matters is actually the std.

TODO: check if the "more frequent updates lead to faster convergence but same terminal loss" phenomena holds in higher H and E.

TODO: use the more frequest updates optimization to train bigger H and E.

Surprisingly, the std doesn't match between the `reset_cache` and the one reported after 20 epochs.

TODO: check the reset cache test every 2 epoch vs reset cache every 1 epoch.

# Feb 12

There's a diff in std between the reset cache test every 2 epoch vs reset cache every 1 epoch, so reset_cache didn't achieve what we have hoped.

Another experiment: run a run with very minimal training vs very minimal training + extensive test, and see if their test loss differs. If not, then it is probably safe.

It doesn't seem like testing should affect the training, since `forward_backward` doesn't set the internals.

Ignore the discrepancy for now.

Scan update_freq for H-500 E-200. First I scan for batch_size.

Maximized for H-500
EMISSIONS=(100 200 300 400 500)
BATCH_SIZES=(256 256 256 256 256)

# Feb 13

Experiments yesterday suggest that higher `update_freq` learns the global minimum by converging below that of lower `update_freq`, but also comes at converging slower.

There is also evidence of a phase transition in switching of the local minimum to the global minimum by varying batch sizes (batch sizes are essentially `update_freq`).

It will be interesting to see if `H-500-E-200` with `update_freq-all` will plateau at the same loss with `update_freq-8,16,32` or if it will go below it.

Will schedule a 24h to pick up from the 10 h mark.

First, the model saved should have the epoch of that save in the filename, instead of the max epoch desired.

Let me edit the file name for the model manually first.

It turns out we didn't save it, because it only gets saved after all epochs are run.

This also means that the file names are all correct.

I will edit the code to introduce checkpointing every x epochs.

# Feb 14

Idea: add time tracking for save and for a general epoch. Done.

Sanity check the outputs.

Get back to building strategy KL divergences.

Now build tools to run from a saved model.

# Feb 15

Copying over notes from the meeting with Eran

### Updates

update_freq and the theory of global vs local learning. (bigger batch size means converge slower, but converge at lower loss).

- Phase transition of test loss using batch sizes

### Feedback

Are the outputs reasonable?

Why more hidden states than emission states work?

Might be more natural to use same number of hidden states. Or more emission states than hidden states.

Now focus on creating the synthetic data.

- Keep the transition matrix but vary the emission probabilities.

### Johnathan

Train script can cut.

Moved from GPT-2 to Llama becuase GPT-2 only uses absolute positioning.

HF implementation of Llama model.

Johnathan used a 10M transformer for 50 emissions.

- success if the transfomer has the transitionary loss for the bigrams.

He used the stationary distribution of Markov chains to generate the starting tokens.

d = 768

number of layers matter more than number of heads.

https://github.com/johnathansun/icll-models

## Today notes

Let me begin by verifying that the model outputs are reasonable. Indeed, they are very much reasonable (see `reasonable.txt`).

Now we can work on getting the Llama set up. That can inform whether we want to use integer or strings for our synthetic data.

I copied over code from Johnathan's repo ([link](https://github.com/johnathansun/icll-models)) and put it in `llm/`. Ran it and it is successful without errors for one GPU.

Submitted another run for 2 GPUs. Somehow 4 GPUs gave this error: `sbatch: error: Batch job submission failed: Requested node configuration is not available`.

2 GPUs are ok. I am not sure whether it is faster.

The main tasks from now is to understand the code by going through it and customize it to my purpose.

# Feb 16

Refactored `train.py`.

# Feb 17

Refactored `llama.py`, `eval.py` and `data.py`.

# Feb 18

Create the dataset using the HMMs. Run it.
Whether we need string tokens or integer tokens. Integers.

Sbatched a H-500-E-100.

Trying to run the data generation, but there's an error saying the model is still on the gpu.

# Feb 19

Keep a remote copy of pomegranate at [https://github.com/jeqcho/pomegranate](https://github.com/jeqcho/pomegranate).

The error now: `TypeError: cannot assign 'torch.FloatTensor' as parameter 'emission_probs' (torch.nn.Parameter or None expected)`

# Feb 20

Fixed all errors. Sbatched a data generation run for H-500-E-100. Also sbatched training runs of H-200-E-200.

# Feb 21

Sbatched a data generation run for H-200-E-200.

Let's train the LLM on the outputs of H-500-E-100.

# Feb 22

We will now generate datasets that permutate the emission matrix. Since we will generate the dataset using batches for efficiency, each batch will be from the same matrix, but it should be fine given that we using `shuffle=True` in the `dataloader`.

First let me add a cli parser for the data generation.

OK I sbatched a 100M sequence data generation for H-200-E-200 with permutations.

I want to generate one with H-500-E-200 so I sbatched a train for that model. TODO generate data with it.

Idea: for non-permutate (or even with permutate), compare how the LLM learns the HMM if we vary H.

# Feb 25

Let's train on the permutated H-200-E-200. But first let's check why the others didn't save. Ok it did save.

We need to generate a larger dataset. Currently it is 100M tokens (using 8 hours). But that only satisfies 100 seconds of trianing (1M tokens/s). We should aim for 5B tokens, which is 83 minutes of training. Naively, that is also 400 hours of GPU time. Let's try to use claude 3.7 to speed things up.

Current benchmark for H-200-E-200-perms is 4m35s for 1M.

1. Change batch size
Seems like we can scale until 4096 and it still works fine. What took 4m is now 20s.

2. Pre-allocate tensors
This reduced 4m to 2m.

These two allows for 20 hours of GPU instead of 400.

Testing it on the 100M H-200-E-200 gives an estimated 1h instead of 8h using 1024.

# Feb 27

On the synthetic data generation, it now reaches 1.7T per file. I have two files so that's 3.4TB. First let's make sure we are nowhere near the lab's quota.
[Handbook](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/storage_and_data_transfer/understanding_storage_options.html) says that the lab has 50 TB. Calling `du -sh /n/netscratch/sham_lab` to get the amount of storage used.

Another concern is that we budgeted 55 hours for each run. 46h passed and 18h to go, so total 64 h. We still need to give it 10h at least, or maybe 15h for the data to save. Let's see if we can give more time to an sbatched run? It is possible to use `scontrol` but access is denied.

I cut loss and will sbatch a new run. I will create 5 parallel copies, each is 1B tokens.

Actually, let's think if we actually need that much. Chincilla says 20 tokens per parameter. Let's see how big is the Llama.

Llama is about 100M parameters. We need about 2T tokens.

I also realize we set the models to `cpu`. I sbatched to compare the new speed after removing that line. Somehow, removing that line makes things runs slower. It is aspect of that currently is going to take one and 70 hours to run five minutes talking while previously it was just 60 hours.

# Feb 28

Suspiciously, the data generation with permutate emissions took only 22h, while without permutate is estimated to be 29h.

For H-200-E-200-2B-perm it took 22h with 30 minutes to save.
For H-200-E-200-2B it estimates 29h.
For H-500-E-200-2B-perm it estimates 25h.
For H-500-E-200-2B it estimates 23h.

I am not sure whether we can trust the output of For H-200-E-200-2B.

Regardless, let's train a LM on this.

Changed the dataloader to use a memorymapped dataset.

# March 1

The LM runs failed. The classic failed because
`RuntimeError: [enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 800,000,000,000 bytes. Error code 12 (Cannot allocate memory)`

The memorymapped run went further and started training. It failed with
`RuntimeError: DataLoader worker (pid(s) 1412867) exited unexpectedly`

The email says its out of memory.

Reduced the chunk size to 100x of the batch size instead of 500. Managed to run.