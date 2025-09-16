from transformers4rec import torch as tr
from transformers4rec.config import transformer_config

# Create BERT4Rec configuration
bert_config = transformer_config.BERT4RecConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=256,
    hidden_dropout_prob=0.3,
    max_sequence_length=50
)
model = tr.BERT4Rec(bert_config)