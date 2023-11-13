import torch
import pandas as pd
import model


### import data 
file_path = 'weatherHistory.csv'

def readCsvToFrame () :
    try :
        dataFrame = pd.read_csv(file_path)
        return dataFrame
    except :
        print("no data was found at " + file_path)
        return

data = readCsvToFrame()


### setting env variables
weather_model = model.Weather_forcast()
weather_model.set_params(data)
lr = 1 # le faire jouer plus tard
optimizer = torch.optim.Adam(weather_model.parameters(), lr)
train_loss = 0
test_loss = 0
test_ind = int(0.8*data.shape[0])


# we train the model with the data retreived from
weather_model.train()
weather_model.set_sample(data, 0, test_ind)
log_probs = weather_model.forward()
loss = -log_probs.mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
train_loss += loss.cpu().data.numpy().item()
print(train_loss)


# We test our model
weather_model.eval()
weather_model.set_sample(data, test_ind, -1)
log_probs = weather_model.forward()
loss = -log_probs.mean()
test_loss += loss.cpu().data.numpy().item()
print(test_loss)

### To Do
"""
    - labélisé ma matrice de transiction (sinon je ne peux pas echantilloné pour déterminé ma marche)
    - 
"""
print("issou")