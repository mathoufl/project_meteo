import torch
import pandas as pd

def log_domain_matmul(log_A: torch.tensor, log_B: torch.tensor):
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]
	log_A_expanded = torch.reshape(log_A, (m,n,1))
	log_B_expanded = torch.reshape(log_B, (1,n,p))
	elementwise_sum = log_A_expanded + log_B_expanded
	out = torch.logsumexp(elementwise_sum, dim=1)
	return out


def maxmul(log_A: torch.tensor, log_B: torch.tensor):
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]
	log_A_expanded = torch.stack([log_A] * p, dim=2)
	log_B_expanded = torch.stack([log_B] * m, dim=0)
	elementwise_sum = log_A_expanded + log_B_expanded
	out1,out2 = torch.max(elementwise_sum, dim=1)
	return out1,out2

def extract_data (data: pd.DataFrame):
	return torch.tensor(data.drop(['Formatted Date', 'Summary', 'Daily Summary'], axis=1).values)

def extract_epoch(data: pd.DataFrame, ind_init: int, ind_end: int) :
	return(data[ind_init : ind_end])