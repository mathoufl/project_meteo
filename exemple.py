import torch
import torch.nn as nn
import torch.optim as optim

class HMM(nn.Module):
    def __init__(self, num_states, num_observations):
        super(HMM, self).__init__()
        self.num_states = num_states
        self.num_observations = num_observations

        # Transition matrix (A) and emission matrix (B)
        self.A = nn.Parameter(torch.rand(num_states, num_states))
        self.B = nn.Parameter(torch.rand(num_states, num_observations))

        # Initial state distribution (pi)
        self.pi = nn.Parameter(torch.rand(num_states))

    def forward(self, observations):
        T = observations.size(0)
        alpha = torch.zeros(T, self.num_states)

        # Initialization
        alpha[0, :] = self.pi * self.B[:, observations[0]]

        # Forward pass
        for t in range(1, T):
            alpha[t, :] = torch.matmul(alpha[t - 1, :], self.A) * self.B[:, observations[t]]

        # Prediction: Probability of observation sequence given the model
        prediction = alpha[-1, :].sum()

        return prediction

# Dummy data (replace this with your actual data loading and preprocessing)
obs_sequence = torch.randint(0, 3, (10,))  # Replace 3 with the number of observation categories
obs_sequence = obs_sequence.long()

# Model instantiation
num_states = 3  # Replace with the number of hidden states
num_observations = 3  # Replace with the number of observation categories
hmm_model = HMM(num_states, num_observations)

# Loss function and optimizer
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss for sequence prediction
optimizer = optim.Adam(hmm_model.parameters(), lr=0.01)

# Training loop (replace this with your actual training loop)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = -hmm_model(obs_sequence)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Example of making predictions
new_observation_sequence = torch.randint(0, 3, (5,)).long()
prediction = hmm_model(new_observation_sequence)
print(f'Predicted Probability: {prediction.item()}')
