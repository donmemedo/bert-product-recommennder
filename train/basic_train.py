from transformers import BertConfig, BertForMaskedLM

config = BertConfig(
    # vocab_size=num_items,  # Your product vocabulary size
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=256,
    hidden_act="gelu",
    # max_position_embeddings=max_seq_length
)
model = BertForMaskedLM(config)