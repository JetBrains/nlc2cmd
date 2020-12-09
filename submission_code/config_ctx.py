from dataclasses import dataclass


@dataclass
class Config:
	batch_size: int = 128
	num_epochs: int = 50
	h_size: int = 1024
	enc_layers: int = 2
	dec_layers: int = 2
	n_heads: int = 16
	d_ff: int = 2048
	src_vocab_size: int = 11000
	tgt_vocab_size: int = 11000
	weight_decay: float = 0
	optimizer: str = 'Adam'
	optimizer_lr: float = 2e-4
	schedule: str = 'ReduceLROnPlateau'
	plateau_factor: float = 0.3
	max_src_len: int = 200
	max_tgt_len: int = 30
	joined_vocab: bool = True
	dropout: float = 0.2