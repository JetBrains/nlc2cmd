from dataclasses import dataclass

@dataclass
class Config:
	batch_size: int = 128
	num_epochs: int = 50
	h_size: int = 512
	n_layers: int = 3
	n_heads: int = 8
	d_ff: int = 2048
	src_vocab_size: int = 8000
	weight_decay: float = 0
	optimizer: str = 'Adam'
	optimizer_lr: float = 2e-4
	schedule: str = 'OneCycleLR'
	max_src_len: int = 35
	joined_vocab: bool = False
	dropout: float = 0.2