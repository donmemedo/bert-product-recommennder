from recbole.quick_start import run_recbole
parameter_dict = {
    'train_neg_sample_args': None,
    'hidden_size': 64,
    'n_layers': 2,
    'n_heads': 2,
    'mask_ratio': 0.2
}
run_recbole(model='BERT4Rec', dataset='your_dataset', config_dict=parameter_dict)