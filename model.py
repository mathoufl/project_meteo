import torch
import pandas as pd
import numpy as np
import random as rd
import util



class TransitionModel(torch.nn.Module):
    N: int
    unnormalized_transition_matrix: torch.nn.Parameter

    def __init__(self):
        super(TransitionModel, self).__init__()


    def set_params(self, N) :
        self.N = N
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(N,N))


    def forward (self, log_alpha) :
        log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

        # Matrix multiplication in the log domain
        out = util.log_domain_matmul(log_transition_matrix, log_alpha.transpose(0,1)).transpose(0,1)
        return out




class EmissionModel(torch.nn.Module):
    N : int
    unnormalized_emission_matrix : torch.nn.Parameter

    def __init__(self):
        super(EmissionModel, self).__init__()


    def set_params(self, N, M):
        self.N = N
        self.M = M
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(N,M)) 


    def forward (self, obervation_raw):
        log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1)
        out = log_emission_matrix[:, obervation_raw].transpose(0,1)
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
    transition_model : TransitionModel
    emission_model : EmissionModel
    unnormalized_state_priors_proba : torch.nn.Parameter


    """
        Temp params for a specific training
    """
    batch : torch.tensor
    batch_labels : pd.DataFrame


    def __init__(self) :
        super(Weather_forcast, self).__init__()

        self.transition_model = TransitionModel()
        self.emission_model = EmissionModel()

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda: self.cuda()


    def set_params(self, data: pd.DataFrame) :
        self.N = len(set(data["Summary"].values))
        self.M = data.drop(["Formatted Date", "Summary", "Daily Summary"], axis=1).shape[1]

        self.transition_model.set_params(self.N)
        self.emission_model.set_params(self.N,self.M)
        self.unnormalized_state_priors_proba = torch.nn.Parameter(torch.randn(self.N))


    def set_sample (self, data: pd.DataFrame, init_ind: int,  end_ind: int) :
        sample_data = data[init_ind : end_ind]
        sample = sample_data.drop(["Formatted Date", "Summary", "Daily Summary"], axis=1)
        sample_filter_rain = sample["Precip Type"] == "rain"
        sample_filter_snow = sample["Precip Type"] == 'snow'
        reverse_sample_filter = sample_filter_rain + sample_filter_snow
        sample[~reverse_sample_filter]["Precip Type"] = 0
        sample[sample["Precip Type"] == 'snow']["Precip Type"] = 2
        sample[sample["Precip Type"] == 'rain']["Precip Type"] = 1
        self.batch = torch.tensor(sample.values)
        self.batch_labels = sample_data["Formatted Date", "Summary", "Daily Summary"]


    def forward(self) :
        # refactor ça pour faire le sampling ? ou passer diréctement un sample ?
        batch_size = self.batch.shape[0]
        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors_proba, dim=0)
        log_alpha = torch.zeros(batch_size, self.M, self.N)
        if self.is_cuda: log_alpha = log_alpha.cuda()

        log_alpha[:, 0, :] = self.emission_model(self.batch[:,0]) + log_state_priors
        for t in range(1, self.M):
            log_alpha[:, t, :] = self.emission_model(self.batch[:,t]) + self.transition_model(log_alpha[:, t-1, :])

        # Select the sum for the final timestep (each x may have different length).
        log_sums = log_alpha.logsumexp(dim=2)
        return log_sums