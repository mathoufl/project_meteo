import torch
import pandas as pd
import model
import util
import random as rd


### import data 
file_path = 'weatherHistory.csv'

def readCsvToFrame () :
    try :
        dataFrame = pd.read_csv(file_path)
        return dataFrame
    except :
        print("no data was found at " + file_path)
        return
    
def format_data() :
    value_mapping = {None: 0, 'snow': 1, 'rain': 2}
    data['Precip Type'] = data['Precip Type'].map(value_mapping)

data = readCsvToFrame()
format_data()


### setting env variables
obs_number = data.shape[1] - 3
raw_number = data.shape[0]
weather_model = model.Weather_forcast(obs_number, raw_number)

lr = 1 # le faire jouer plus tard
optimizer = torch.optim.Adam(weather_model.parameters(), lr)
train_loss = 0
test_loss = 0
test_ind = int(0.8*data.shape[0])
ind = 0


# we train the model with the data retreived from
weather_model.train()
while ind <= test_ind :
    random_ind_end = ind + rd.randint(0, int(0.001*test_ind)) 
    ind_end = random_ind_end if random_ind_end < test_ind else test_ind
    epoch = util.extract_epoch(data, ind, ind_end)
    ind = ind_end
    log_probs = weather_model.forward(epoch)
    loss = -log_probs.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss += loss.cpu().data.numpy().item()
    print(train_loss)


# We test our model
weather_model.eval()
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