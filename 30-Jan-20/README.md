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