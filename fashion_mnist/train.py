import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms, datasets

from model import FashionCNN


DATA_PATH = '../data'
NUM_CLASSES = 10
BATCH_SIZE = 100
NUM_EPOCH = 1
LEARNING_RATE = 0.001

def train_model(model, train_loader, device):
    model.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    count = 0
    loss_list = []
    iteration_list = []
    for epoch in range(NUM_EPOCH):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
        
            # train = Variable(images.view(100, 1, 28, 28))
            # labels = Variable(labels)
             
            outputs = model(images)
            loss = loss_func(outputs, labels)
            
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

def main():
    #Загружаем данные, обучаем модель и сохраняем на диск
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # image, label = next(iter(train_set))
    # print(label, image.shape)
    # plt.imshow(image.squeeze(), cmap="gray")
    # plt.savefig("./example.png")

    model = FashionCNN(num_classes=NUM_CLASSES)
    model.to(device)

    loss_list, iteration_list = train_model(model, train_loader, device)

    torch.save(model.state_dict(), './trained_weights_FashionCNN.pth')

    # plt.plot(iteration_list, loss_list)
    # plt.xlabel("No. of Iteration")
    # plt.ylabel("Loss")
    # plt.title("Iterations vs Loss")
    # plt.savefig("./training_loss.png")

if __name__ == '__main__':
    main()