import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms, datasets

from model import FashionCNN


BATCH_SIZE = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader):
    model.train()

    Loss_func = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 5
    count = 0

    loss_list = []
    iteration_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
        
            # train = Variable(images.view(100, 1, 28, 28))
            # labels = Variable(labels)
             
            outputs = model(images)
            loss = Loss_func(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            count += 1

            if not (count % 50):
                loss_list.append(loss.data)
                iteration_list.append(count)

            if not (count % 500):
                print("Iteration: {}, Loss: {}".format(count, loss.data))

    return loss_list, iteration_list



if __name__ == '__main__':
    #Загружаем данные, обучаем модель и сохраняем на диск

    train_set = datasets.FashionMNIST(root='./', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # image, label = next(iter(train_set))
    # print(label, image.shape)
    # plt.imshow(image.squeeze(), cmap="gray")
    # plt.savefig("./example.png")

    model = FashionCNN()
    model.to(device)

    loss_list, iteration_list = train_model(model, train_loader)

    torch.save(model.state_dict(), './trained_weights_FascionCNN.pth')

    # plt.plot(iteration_list, loss_list)
    # plt.xlabel("No. of Iteration")
    # plt.ylabel("Loss")
    # plt.title("Iterations vs Loss")
    # plt.savefig("./training_loss.png")