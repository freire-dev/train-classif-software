import torch
from torch import nn, optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

torch.manual_seed(123)

# Code block to build classif. model

classificador = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
                              nn.ReLU(),
                              nn.BatchNorm2d(num_features=32),
                              # (64 - 3 + 1) / 1 = 62x62
                              nn.MaxPool2d(kernel_size=2),
                              # 31x31
                              nn.Conv2d(32, 32, 3),
                              nn.ReLU(),
                              nn.BatchNorm2d(32),
                              # (31 - 3 + 1) / 1 = 29x29
                              nn.MaxPool2d(2),
                              #  14x14
                              nn.Flatten(),
                              # 6272 -> 128 -> 128 -> 1
                              nn.Linear(in_features=14*14 * \
                                        32, out_features=128),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(128, 128),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(128, 1),
                              nn.Sigmoid())

criterion = nn.BCELoss()
optimizer = optim.Adam(classificador.parameters())

data_dir_train = './data'
data_dir_test = './data'

transform_train = transforms.Compose(
    [
     transforms.Resize([64, 64]),
     transforms.RandomHorizontalFlip(),
     transforms.RandomAffine(degrees=7, translate=(0, 0.07), shear=0.2, scale=(1, 1.2)),
     transforms.ToTensor()
    ]
)

transform_test = transforms.Compose(
    [
     transforms.Resize([64, 64]),
     transforms.ToTensor()
    ]
)

train_dataset = datasets.ImageFolder(data_dir_train, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle=True)

test_dataset = datasets.ImageFolder(data_dir_test, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle=True)

#Code block to build training model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classificador.to(device)

def training_loop(loader, epoch):
    running_loss = 0.
    running_accuracy = 0.
    
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()        
        outputs = classificador(inputs)
        
        loss = criterion(outputs, labels.float().view(*outputs.shape))
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.tensor([1 if output > 0.5 else 0 for output in outputs]).to(device)
        
        equals = predicted == labels.view(*predicted.shape)
        
        accuracy = torch.mean(equals.float())
        running_accuracy += accuracy
                   
        # Imprimindo os dados referentes a esse loop
        print('\rÉPOCA {:3d} - Loop {:3d} de {:3d}: perda {:03.2f} - precisão {:03.2f}'.format(epoch + 1, i + 1, len(loader), loss, accuracy), end = '\r')
        
    # Imprimindo os dados referentes a essa época
    print('\rÉPOCA {:3d} FINALIZADA: perda {:.5f} - precisão {:.5f}'.format(epoch + 1, running_loss/len(loader), 
                    running_accuracy/len(loader)))

for epoch in range(10):
    print('Treinando...')
    training_loop(train_loader, epoch)
    classificador.eval()
    print('Validando...')
    training_loop(test_loader, epoch)
    classificador.train()

#Run image classif.

def classificar_imagem(fname):

  imagem_teste = Image.open('./data_teste_para_classificar' + '/' + fname)

  imagem_teste = imagem_teste.resize((64, 64))
  imagem_teste = np.array(imagem_teste.getdata()).reshape(*imagem_teste.size, 3)
  imagem_teste = imagem_teste / 255
  imagem_teste = imagem_teste.transpose(2, 0, 1)
  imagem_teste = torch.tensor(imagem_teste, dtype=torch.float).view(-1, *imagem_teste.shape)

  classificador.eval()
  imagem_teste = imagem_teste.to(device)
  output = classificador.forward(imagem_teste)
  print(f'output: {output}')
  if output > 0.5:
    output = 1
  else:
    output = 0

  idx_to_class = {value: key for key, value in test_dataset.class_to_idx.items()}

  return print(f'Previsão: {output} = {idx_to_class[output]}')