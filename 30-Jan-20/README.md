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