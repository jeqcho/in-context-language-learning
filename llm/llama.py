import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel
import yaml


class Llama(nn.Module):
    """
    Custom Llama model that wraps Hugging Face's LlamaModel with an additional linear layer.
    """

    def __init__(self, config: dict):
        """
        Initialize the Llama model.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()
        self.cfg = config
        self._unpack_dict(self.cfg)
        self.model_params = self.cfg["model"]

        # Compute vocabulary size from configuration values.
        self.vocab_size = self.cfg["markov"]["high_idx"] - self.cfg["markov"]["low_idx"]

        # Set default activation and position parameters.
        self.activation_function = "gelu"
        self.n_positions = self.model_params["seq_len"] + 1
        self.use_cache = False

        # Create the Hugging Face Llama configuration.
        llama_config = LlamaConfig(
            vocab_size=self.vocab_size,
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
            output_hidden_states=self.output_hidden_states,
        )

        # Initialize the Llama model and a linear output layer.
        self.llama_model = LlamaModel(llama_config)
        self.linear = nn.Linear(self.hid_dim, self.vocab_size)

    def _unpack_dict(self, config) -> None:
        """
        Unpack the configuration dictionary and set attributes for each key.

        Args:
            config (dict): Nested configuration dictionary.
        """
        for section_key in config.keys():
            for key, value in config[section_key].items():
                setattr(self, key, value)

    def forward(
        self,
        x: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor (token IDs).
            output_hidden_states (bool): If True, returns the hidden states.
            output_attentions (bool): If True, returns the attention weights.

        Returns:
            torch.Tensor: Logits output from the linear layer.
            Optionally, a tuple with attentions or hidden states if the corresponding flags are set.
        """
        outputs = self.llama_model(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.linear(outputs.last_hidden_state)

        if output_attentions:
            return logits, outputs.attentions
        if output_hidden_states:
            return logits, outputs
        return logits

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and linear layer weights to the specified directory.

        Args:
            save_directory (str): Directory to save the model.
        """
        self.llama_model.save_pretrained(save_directory)
        torch.save(self.linear.state_dict(), f"{save_directory}/linear_layer.bin")

    @classmethod
    def from_pretrained(cls, config: dict, save_directory: str):
        """
        Load the model and linear layer weights from a pretrained directory.

        Args:
            config (dict): Configuration dictionary.
            save_directory (str): Directory where the model is saved.

        Returns:
            Llama: The loaded model instance.
        """
        model = cls(config)
        model.llama_model = LlamaModel.from_pretrained(save_directory)
        model.linear.load_state_dict(torch.load(f"{save_directory}/linear_layer.bin"))
        return model

    def get_num_parameters(self) -> int:
        """
        Count the total number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage:
if __name__ == "__main__":
    config_path = (
        "/n/holyscratch01/sham_lab/Users/jlsun/icll-models/markov/wandb_config.yaml"
    )
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model = Llama(config).to("cuda")
    print(f"Number of trainable parameters: {model.get_num_parameters()}")
