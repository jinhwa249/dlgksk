import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('python/car_evaluation.csv')
dataset.head()

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot(kind='pie', autopct='%0.05f%%',
colors=['lightblue', 'lightgreen', 'pink'], explode=(0.05, 0.05, 0.05, 0.05))
# plt.show()

categorical_columns = ['price', 'maint', 'doors', 'persons', 'iug_capacity', 'safety']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')
    
price= dataset['price'].cat.codes.values
maint= dataset['maint'].cat.codes.values
doors= dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
categorical_data[:10]

categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
categorical_data[:10]

outputs = pd.get_dummies(dataset.output)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()

print(categorical_data.shape)
print(outputs.shape)

categorical_columns_sizes = [len(dataset[column].cat.categories) for column in
                             categorical_columns]
categorical_columns_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in
                              categorical_columns_sizes]
print(categorical_embedding_sizes)

total_records =1728
test_records = int(total_records * 2)

categorical_train_data = categorical_data[:total_records - test_records]
categorical_test_data = categorical_data[:total_records - test_records: total_records]
train_outputs = outputs[:total_records - test_records]
test_outputs = outputs[:total_records - test_records: total_records]

print(len(categorical_train_data))
print(len(train_outputs))
print(len(categorical_test_data))
print(len(test_outputs))

class Model(nn. Module):
    def __init__(self, embedding_size, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni,
                                            nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        
        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols 
        
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.Batchwormid(i))
            all_layers.append(nn.Dropout(p))
            input_size = i
            
        all_layers.append(nn. Linear(layers[-1], output_size))
        self. layers = nn.Sequential(*all_layers)
    def forward(self, x_categorical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))

        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x
    
model = Model(categorical_embedeing_sizes, 4, [200,100,50], p=0.4)
print(model)
    
loss_function = nn.CrossEntropyLoss()
optmizer = torch.optim.Adam(midel.parameters(), lr=0.001)

if torch.cuda_is_available():
    device = torch.device('cude')
eles:
    device = torch.device('cpu')
    
epochs = 500
aggregsted_losses = []
train_outputs = train_outputs.to(device=device, dtype=torch=torch.int64)
for i in range(epochs):
    i += 1
    y_pred = madel(categor_train_data)
    single_losses = loss_function(y_pred, train_outputs)
    aggreated_losses.append(single_loss)
    
    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        
        optimizer.zero_grad()
        single_loss.backward()
        optimizer.step()
        
print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
epochs = 500
aggregated_lossea = []
train_outputs = train_outputs.to(device=device, dtype=torch.int64)
for i in range(epochs):
i += 1
y_pred = model(categorical _train_data).to(device)
single_loss = loss function(y. pred train outputs)
aggregated_losses.append(single_loss)


if i%25 == 1:
    print(f'epoch: {i:3} loss: {single loss.item():10.8f}')

optimizer.zero_grad()
single_loss.backward()
optimizer.step()

print(f'epoch: {i:3} loss: {single loss.item():10.10f}')

test_outputs = test_outputs.to(device=device, dtype=torch.int64)
with torch.no_grad():
    y_val = madel(categoricel_test_data)
    loss = loss_function(y_val, test_out_outputs)
print('Loss: {loss: 8f}')

print(y_val[:5])

y_val = np.argmax(y_val, axis=1)
pront(y_val[:5])

from sklearn.metrics import classification report, confusion matrix, accuracy_score
print(confusion_matrix(test_outputs,yval) )
print(classification report(test_outputs,y_val))
print(accuracyscore(test outputs, y-val))