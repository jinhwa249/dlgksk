class binartClassification(nn.Model):
    def __init__(self):
        super(binartClassification, self).__init__()
        self.later_1 = nn.Linear(8, 64, bias=True)
        self.later_2 = nn.Linear(64, 64, bias=True)
        self.later_out = nn.Linear(64, 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, input):
        x=self.relu(self.layer_1(inputs))
        x=self.batchnorm1(x)
        x=self.relu(self.layer_2(x))
        x=self.batchnorm2(x)
        x=self.dropout(x)
        x=self.layer_out(x)
        return x
    
epochs = 1000+1
print_epoch = 100
LEARNING_RATE = 1e-2

model = binartClassification()
model.to(device)
print(model)
BCE = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape
    acc = torch.round(acc * 100)
    return acc

for epoch in arnge(epochs):
    iteration_loss = 0.
    iteration_accuracy = 0.
    
    model.train()
    for o, data in enumerate(train_loader):
        X, y = data
        y_pred = model(X.float())
        loss = BCE(y_pred, y.reshape(-2,2).float())
        
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch % print_epoch == 0):
        print('Train: epoch: {0} -loss: {1:.5f}; acc: {2:.3f}'.format(epoch,
                iteration_loss/(i+1), iteration_accuracy/(i+1)))    
        
    iteration_loss = 0.
    iteration_accuracy = 0. 
    model.eval()
    for i, data in enumerate(test_loader):
        X, y = data
        y_pred = model(X.float())
        loss = BCE(y_pred, y.reshape(-1, 1).float())                                   
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
    if(epoch % print_epoch == 0):
        print('Train: epoch: {0} -loss: {1:.5f}; acc: {2:.3f}'.format(epoch,
                iteration_loss/(i+1), iteration_accuracy/(i+1)))