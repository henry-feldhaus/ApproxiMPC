import json
import os

import joblib
import torch
import torch.nn as nn


class F1TenthSequenceModel(nn.Module):
	def __init__(
		self,
		input_dim,
		rnn_hidden_dim,
		output_dim,
		model_type="LSTM",
		num_rnn_layers=2,
		use_embedding=True,
		embedding_dim=500,
		fc_layer_sizes=None,
		dropout=0.2,
	):
		super(F1TenthSequenceModel, self).__init__()

		if fc_layer_sizes is None:
			fc_layer_sizes = [256, 128, 32]

		self.model_type = model_type.upper()
		self.use_embedding = use_embedding

		if self.use_embedding:
			self.embedding = nn.Sequential(
				nn.Linear(input_dim, embedding_dim),
				nn.ReLU(),
			)
			rnn_input_dim = embedding_dim
		else:
			self.embedding = None
			rnn_input_dim = input_dim

		if self.model_type == "LSTM":
			self.rnn = nn.LSTM(
				input_size=rnn_input_dim,
				hidden_size=rnn_hidden_dim,
				num_layers=num_rnn_layers,
				batch_first=True,
				dropout=dropout if num_rnn_layers > 1 else 0,
			)
		elif self.model_type == "GRU":
			self.rnn = nn.GRU(
				input_size=rnn_input_dim,
				hidden_size=rnn_hidden_dim,
				num_layers=num_rnn_layers,
				batch_first=True,
				dropout=dropout if num_rnn_layers > 1 else 0,
			)
		else:
			self.rnn = nn.RNN(
				input_size=rnn_input_dim,
				hidden_size=rnn_hidden_dim,
				num_layers=num_rnn_layers,
				batch_first=True,
				dropout=dropout if num_rnn_layers > 1 else 0,
			)

		fc_modules = []
		current_input_size = rnn_hidden_dim
		for hidden_size in fc_layer_sizes:
			fc_modules.append(nn.Linear(current_input_size, hidden_size))
			fc_modules.append(nn.ReLU())
			current_input_size = hidden_size

		fc_modules.append(nn.Linear(current_input_size, output_dim))
		self.fc = nn.Sequential(*fc_modules)

	def forward(self, x):
		batch_size, seq_len, _ = x.size()

		if self.use_embedding:
			x_reshaped = x.view(batch_size * seq_len, -1)
			embedded = self.embedding(x_reshaped)
			rnn_in = embedded.view(batch_size, seq_len, -1)
		else:
			rnn_in = x

		if self.model_type == "LSTM":
			rnn_out, _ = self.rnn(rnn_in)
		else:
			rnn_out, _ = self.rnn(rnn_in)

		final_timestep_out = rnn_out[:, -1, :]
		return self.fc(final_timestep_out)


def load_model_from_artifacts(
	config_path,
	model_path,
	input_scaler_path,
	target_scaler_path,
	device="cpu",
):
	"""Load model and scalers from explicit artifact file paths."""
	with open(config_path, "r") as f:
		cfg = json.load(f)

	model_cfg = cfg.get("model_config", {})
	model = F1TenthSequenceModel(**model_cfg).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()

	input_scaler = joblib.load(input_scaler_path)
	target_scaler = joblib.load(target_scaler_path)

	return model, input_scaler, target_scaler, cfg


def load_model_bundle_from_config(config_path, device="cpu"):
	"""Load model and scalers using one config json path plus referenced artifact filenames."""
	with open(config_path, "r") as f:
		cfg = json.load(f)

	candidate_base_dir = cfg.get("artifact_dir")
	if candidate_base_dir and os.path.isdir(candidate_base_dir):
		base_dir = candidate_base_dir
	else:
		# Keep loading robust when configs were moved across machines/workspaces.
		base_dir = os.path.dirname(config_path)
	model_path = os.path.join(base_dir, cfg["state_dict_file"])
	input_scaler_path = os.path.join(base_dir, cfg["input_scaler_file"])
	target_scaler_path = os.path.join(base_dir, cfg["target_scaler_file"])

	return load_model_from_artifacts(
		config_path=config_path,
		model_path=model_path,
		input_scaler_path=input_scaler_path,
		target_scaler_path=target_scaler_path,
		device=device,
	)
