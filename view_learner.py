import torch
from torch.nn import Sequential, Linear, ReLU
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli

class ViewLearner(torch.nn.Module):
	def __init__(self,encoder, mlp_edge_model_dim=32):
		super(ViewLearner, self).__init__()

		self.encoder = encoder
		# self.input_dim = self.encoder.out_node_dim
		self.input_dim = mlp_edge_model_dim
		self.mlp_edge_model = Sequential(
			Linear(self.input_dim*2, mlp_edge_model_dim),
			ReLU(),
			Linear(mlp_edge_model_dim, 1)
		).cuda()
		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)
	def build_prob_neighbourhood(self, edge_wight, temperature=0.1):
		attention = torch.clamp(edge_wight, 0.01, 0.99)

		weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(attention.device),
                                                     probs=attention).rsample()
		import random
		eps = 0.0
		mask = (weighted_adjacency_matrix > eps).detach().float()
		weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)
		return weighted_adjacency_matrix
	# def forward(self, x, edge_index):
	def forward(self, x, edge_index, norm_adjacent_matrix):

		# node_emb,_ = self.encoder(x, norm_adjacent_matrix)
		node_emb,_ = self.encoder(norm_adjacent_matrix, x)

		src, dst = edge_index[0], edge_index[1]
		emb_src = node_emb[src]
		emb_dst = node_emb[dst]

		edge_emb = torch.cat([emb_src, emb_dst], 1)
		# print("edge_emb size:", edge_emb.size())
		edge_logits = self.mlp_edge_model(edge_emb)
		temperature = 1.0
		bias = 0.0 + 0.0001  # If bias is 0, we run into problems
		eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
		gate_inputs = torch.log(eps) - torch.log(1 - eps)
		gate_inputs = gate_inputs.cuda()
		gate_inputs = (gate_inputs + edge_logits) / 1.0
		edge_wight = torch.sigmoid(gate_inputs).squeeze().detach()
		adj = self.build_prob_neighbourhood(edge_wight, temperature=0.9)
		return node_emb, adj