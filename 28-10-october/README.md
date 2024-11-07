# 10 October
Implement the word-level tokenizer.

First I will count words by frequency and choose top 500 words, and then subset for stories that uses this 500 words. (I will also build a histogram to see the number of stories for the number of words we chose)

# 11 October
If we use a 60k parameter model, roughly 100k tokens needed to train. Rule of thumb is one order of magnitude bigger.
Currently with a 600-word tokenizer, we get 221k tokens from 827 stories.

# 17 October
I think we can start with what we have.
Our dataset:
- Number of stories: 827
- Max length: 438
- Number of unique words: 583
- Total tokens (without padding): 221,921

Here are tasks
- Fit a suite of HMMs
- Evaluate how good the HMMs are

Once that is done, we can begin with LLMs.

Copying here two papers that we previously identified to be useful for HMMs
1. [Scaling Hidden Markov Language Models](https://arxiv.org/pdf/2011.04640)
2. [On Limitation of Transformer for Learning HMMs](https://arxiv.org/pdf/2406.04089)


# 18 October
The HMMs are fitted in time closely matching that of prediction, nicely under the 50% buffer.
Only 300 components fit without degenerate solutions, since 400 was fitting 392799 free scalar parameters with only 362226 data points.

Today we evaluate the HMMs.
The 300 and 400 models fail to include spaces between words.
This can be a critical first-test for all future models.

After some thinking, I realize that 300 states are not enough to encode the thoughts that we want. In transformers, those thoughts are encoded in a smooth 256-dimensional space. We might be able to approximate this if we use larger spaces and states. Then we need more tokens. We can go to BPE or character-level tokenization (but this also means we need a larger model for a smaller context window).

Another point is that maybe the model should learn syntax or grammar before meaning. So it makes sense to split sentences up and choose a subset of sentences instead of subset of stories. This will likely make the training set larger.

Next time, I will proceed with
- testing out the 700 model.
- checking if the space token is swapped with the also token (which occurs almost every other word)
- Split stories into sentences
- Rerun the training using those sentences (this also gives a fair comparison to the >300 models)
- Look into using Very Large HMMs.

# 24 October
Let's check if the space token is swapped with the also token.
Conclusion: the hmm already has this baked within, so if anything, it is the hmm training

# move to notion