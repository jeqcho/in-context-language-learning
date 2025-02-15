# %%
import torch as t
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel
import yaml

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self._unpack_dict(self.cfg)
        self.model_params = self.cfg['model']
        self.vocab_size = self.cfg['markov']['high_idx'] - self.cfg['markov']['low_idx']
        self.activation_function= 'gelu'
        self.n_positions = self.model_params['seq_len'] + 1
        self.use_cache=False
        
        config = LlamaConfig(vocab_size=self.vocab_size,
                             hidden_size=self.hid_dim,
                             intermediate_size=self.mlp_ratio * self.hid_dim,
                             num_hidden_layers=self.n_layer,
                             num_attention_heads=self.n_head,
                             hidden_act=self.activation_function,
                             max_position_embeddings=self.n_positions,
                             rope_scaling=self.rope_scaling,
                             attention_bias=self.attn_bias,
                             attention_dropout=self.attn_pdrop,
                             mlp_bias=self.mlp_bias,
                             output_hidden_states=self.output_hidden_states)
        
        self.LlamaModel = LlamaModel(config)
        self.lin2 = nn.Linear(self.hid_dim, self.vocab_size)

    def _unpack_dict(self, d):
        for section_key in d.keys():
            for k, v in d[section_key].items():
                setattr(self, k, v)

    def forward(self, x, output_hidden_states=False, output_attentions=False):
        outputs = self.LlamaModel(input_ids=x, attention_mask=t.ones_like(x), output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        hidden = outputs.last_hidden_state
        last = self.lin2(hidden)
        if output_attentions:
            return last, outputs.attentions
        elif output_hidden_states:
            return last, outputs
        else:
            return last
    
    def save_pretrained(self, save_directory):
        # Save GPT2 model
        self.LlamaModel.save_pretrained(save_directory)
        # Save the linear layer separately
        t.save(self.lin2.state_dict(), f"{save_directory}/linear_layer.bin")

    @classmethod
    def from_pretrained(cls, config, save_directory):
        # Create a new instance of the class
        model = cls(config)
        model.LlamaModel = LlamaModel.from_pretrained(save_directory)
        # Load the linear layer state dict
        model.lin2.load_state_dict(t.load(f"{save_directory}/linear_layer.bin"))
        return model
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# %%

# config_path = '/n/holyscratch01/sham_lab/Users/jlsun/icll-models/markov/wandb_config.yaml'

# with open(config_path, 'r') as file:
#     config = yaml.safe_load(file)

# model = Llama(config).to('cuda')
# %%