#!/usr/bin/env python
# coding: utf-8

# This notebook trains a range of HMM with fixed 100 emission states (e) and 100 context length (L). We train for 200, 400, 800, 1600 hidden states (h).
# 
# Input file:
# 
# `/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-K-train.txt`
# `/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-K-test.txt`
# 
# and the tokenizer
# 
# `/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/tokenizer.json`
# 
# Output file:
# 
# `/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/TinyStories-e-100-L-100-h-{h}.pkl`

# In[22]:


from hmmlearn import hmm
import pickle
import argparse

def train_hmm(num_emission, num_hidden_states, data, lengths, verbose=False):
    model = hmm.CategoricalHMM(n_components=num_hidden_states, n_features=num_emission, verbose=verbose)
    
    model.fit(data, lengths)
    
    return model


# In[14]:


# read the data and divide it into chunks of length context length

def get_hmm_train_data(num_emission, context_length):
    train_data_file_name = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-{num_emission}-train.txt"
    with open(train_data_file_name, 'r') as file:
        lines = file.readlines()
    
    # put all lines on one line
    text = ' '.join([line.strip() for line in lines])
    
    # parse as tokens
    tokens = [int(token) for token in text.split()]
    
    # get chunks of size context length
    chunks = []
    idex = 0
    while idex+context_length < len(tokens):
        chunks.append(tokens[idex:idex+context_length])
        idex += context_length
    
    # for above, note that it is possible that the final bits of the text will be cut off
    
    # the lengths are constant
    lengths = [context_length] * len(chunks)
    
    # preprocess each chunk
    # for the hmm, we need to wrap each emission with [] to show it is 1D
    # and concat all data together
    new_chunks = []
    for chunk in chunks:
        new_chunk = [[token] for token in chunk]
        new_chunks += new_chunk
    
    chunks = new_chunks
    del new_chunks
    
    # chunks is now an array of singletons
    
    return chunks, lengths


# In[15]:


# combinations

class ModelParams:
    L = None
    h = None
    e = None
    
    def __init__(self, L, h, e):
        self.L = L
        self.h = h
        self.e = e
        
    def __str__(self):
        return f"ModelParams(L={self.L}, h={self.h}, e={self.e})"

def main():
    parser = argparse.ArgumentParser(description='Train HMM with varying hidden states.')
    parser.add_argument('index', type=int, help='Index of the hidden state configuration to use')
    args = parser.parse_args()
    index = args.index
    print(index)
    return
    
    hs = [200, 400, 600, 800]

    combinations = [ModelParams(L=100, h=h, e=100) for h in hs]


    # In[16]:


    # test out the chosen combination
    simple_model_params = combinations[index]
    str(simple_model_params)


    # In[17]:


    simple_train_data = get_hmm_train_data(num_emission=simple_model_params.e, context_length=simple_model_params.L)

    data, lengths = simple_train_data


    # In[ ]:


    simple_model = train_hmm(
        num_emission=simple_model_params.e, num_hidden_states=simple_model_params.h, data=data, lengths=lengths, verbose=True
    )


    # In[ ]:


    # save the model

    file_name = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/hmm-L-{simple_model_params.L}-h-{simple_model_params.h}-e-{simple_model_params.e}.txt"

    with open(file_name, "wb") as file:
        pickle.dump(simple_model, file)


if __name__ == "__main__":
    main()