import torch
import pandas as pd
import numpy as np
import random as rd
import util



def forward_trans (self, log_alpha) :
    log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)
    out = util.log_domain_matmul(log_transition_matrix, log_alpha.transpose(0,1)).transpose(0,1)
    return out



def forward_em (self, hidden_state, observation_raw):
    print(observation_raw)
    log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1)
    out = log_emission_matrix[observation_raw, :].transpose(0,1)
    return out


class Weather_forcast (torch.nn.Module) :
    """
        Model's parametres
    """
    N : int # nuumber of possible value taken by the states
    M : int # number of obesrvations fildes possible
    isCuda : bool
 

    """
        Parametres to train
    """
    transition_model : torch.nn.Parameter
    emission_model : torch.nn.Parameter


    """
        Temp params for a specific training
    """
    batch : torch.tensor


    def __init__(self, feature_number: int, state_number: int) :
        self.N = feature_number
        self.M = state_number
        super(Weather_forcast, self).__init__()

        self.transition_model = torch.nn.Parameter(torch.randn(self.N,self.N))
        self.emission_model = torch.nn.Parameter(torch.randn(self.N,self.M)) 

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda: self.cuda()


    def inspect_model(self):
        print(self.transition_model.unnormalized_transition_matrix)
        print(self.emission_model.unnormalized_emission_matrix)
        print(self.unnormalized_state_priors)


    def forward(self, inputs: pd.DataFrame) :
        formated_inputs = util.extract_data(inputs)
        batch_size = formated_inputs.shape[0]
        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
        log_alpha = torch.zeros(batch_size, self.M)
        if self.is_cuda: log_alpha = log_alpha.cuda()

        log_alpha[0, :] = self.emission_model(formated_inputs[0, :]) + log_state_priors
        for t in range(1, self.M):
            log_alpha[:, t, :] = self.emission_model(formated_inputs[:,t]) + self.transition_model(log_alpha[:, t-1, :])

        # Select the sum for the final timestep (each x may have different length).
        log_sums = log_alpha.logsumexp(dim=2)
        return log_sums