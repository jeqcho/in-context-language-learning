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