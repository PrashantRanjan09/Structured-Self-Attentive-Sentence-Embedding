
import torch
import keras
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from keras.preprocessing import sequence
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pack_padded_sequence



top_words = 10000
learning_rate =0.001
max_seq_len = 200
emb_dim = 300
batch_size=500
u=64
da = 32
r= 16


from keras.datasets import imdb
(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words = top_words)
x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_seq_len)


xtrain=[]
for item in x_train:
    item = np.array(item)
    if item.dtype =='O':
        pass
    else:
        xtrain.append(item)

xtest=[]
for item in x_test:
    item = np.array(item)
    if item.dtype =='O':
        pass
    else:
        xtest.append(item)


xtrain = np.array(xtrain, dtype = [('O', np.int)]).astype(np.int)
xtest = np.array(xtest, dtype = [('O', np.int)]).astype(np.int)


class Imdb_train(Dataset):

    def __init__(self):
        self.len = xtrain.shape[0]
        self.x_data_train = torch.from_numpy(x_train)
        self.x_data_train = self.x_data_train.type(torch.LongTensor)
        print(self.x_data_train.size())
        self.y_data_train = torch.from_numpy(y_train)
        self.y_data_train = self.y_data_train.type(torch.FloatTensor)
        self.y_data_train = self.y_data_train.view(-1,1)
        print(self.y_data_train.size())


    def __getitem__(self,index):
        return self.x_data_train[index],self.y_data_train[index]

    def __len__(self):
        return self.len


class Imdb_test(Dataset):

    def __init__(self):
        self.len = xtest.shape[0]
        self.x_data_test = torch.from_numpy(x_test)
        self.x_data_test = self.x_data_test.type(torch.LongTensor)
        print(self.x_data_test.size())
        self.y_data_test = torch.from_numpy(y_test)
        self.y_data_test = self.y_data_test.type(torch.FloatTensor)
        self.y_data_test = self.y_data_test.view(-1,1)
        print(self.y_data_test.size())


    def __getitem__(self,index):
        return self.x_data_test[index],self.y_data_test[index]

    def __len__(self):
        return self.len

dataset_imdb_train = Imdb_train()
train_loader = DataLoader(dataset = dataset_imdb_train,
                          batch_size = 500,
                          shuffle = True)

dataset_imdb_test = Imdb_test()
test_loader = DataLoader(dataset = dataset_imdb_test,
                          batch_size = 500,
                          shuffle = True)



class SelfAttentiveModel(nn.Module):

    def __init__(self,top_words=top_words,emb_dim=emb_dim,max_seq_len=max_seq_len,u=u,da=da):
        super(SelfAttentiveModel,self).__init__()
        self.embedding = nn.Embedding(top_words,emb_dim)
        self.bilstm = nn.LSTM(input_size = emb_dim,hidden_size = u,batch_first=True,bidirectional=True)
        self.lin1 = nn.Linear(2*u,da)
        self.lin2 = nn.Linear(da,r)
        self.lin3 = nn.Linear(r*2*u,1)

    def forward(self,x):
        out = self.embedding(x)
        out_lstm,_ = self.bilstm(out)
        out = self.lin1(out_lstm)
        out = F.tanh(out)
        out = self.lin2(out)
        out_A = F.softmax(out,dim=0)
        temp1 = out_A.permute(0,2,1)
        temp2 = out_lstm
        out = torch.bmm(temp1,temp2) # AH
        out = out.view(500,16*128)
        out = self.lin3(out)
        out = F.sigmoid(out)

        return out

model = SelfAttentiveModel()

print(model)

criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(5):
    total = 0
    n_batches = 0
    correct = 0

    for i,data in enumerate(train_loader):

        inputs,labels = data
        labels = labels.type(torch.FloatTensor)
        inputs,labels = Variable(inputs),Variable(labels)
        #print(inputs)
        #print(labels)
        outs = model(inputs).type(torch.FloatTensor)
        output = outs.round()

        loss = criterion(outs,labels)

        total += labels.size(0)
        correct += (output == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        print('epoch: ',epoch+1,'batch :',i,'loss :',loss.data[0])

        print('Accuracy of the network till this batch: %d %%' % (100 * correct / total))
    print('Accuracy of the network till end of epoch : %d %%' % (100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data_test in test_loader:
        inputs_test, labels_test = data_test
        outs = model(inputs_test)
        outputs = outs.round()
        #outputs = outputs.type(torch.FloatTensor)
        total += labels_test.size(0)
        #print(type(predicted))
        #print(type(labels_test))
        correct += (outputs == labels_test).sum().item()

print('Accuracy of the network on the 25000 test inputs: %d %%' % (
    100 * correct / total))
