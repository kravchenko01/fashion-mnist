import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms, datasets

from model import FashionCNN


BATCH_SIZE = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

def test_model(model, test_loader):
    # Testing the model   
    model.eval()

    correct = 0
    total = 0

    class_correct = [0. for _ in range(10)]
    total_correct = [0. for _ in range(10)]

    predicted_lst = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # test = Variable(images)
            outputs = model(images)

            predicted = torch.max(outputs, 1)[1].to(device)
            predicted_lst += list(predicted.numpy())
            c = (predicted == labels).squeeze()

            correct += (predicted == labels).sum()
            total += len(labels)
            
            for i in range(BATCH_SIZE):
                label = labels[i]
                class_correct[label] += c[i].item()
                total_correct[label] += 1
            

    metrics_dct = {}
    metrics_dct['Accuracy'] = (correct / total).item()

    for i in range(10):
        metrics_dct['Accuracy of {}:'.format(output_label(i))] = class_correct[i] / total_correct[i]
    
    return metrics_dct, predicted_lst

if __name__ == '__main__':
    # считываем с диска модель, загружаем валидационный датасет,
    # предсказываем моделью ответы для этих данных, записываем ответы на диск в .csv файл,
    # выводим в stdout (`print`) необходимые метрики на этом датасете.

    test_set = datasets.FashionMNIST(root='./', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = FashionCNN()
    model.load_state_dict(torch.load('./trained_weights_FashionCNN.pth'))
    model.to(device)

    metrics_dct, predicted_lst = test_model(model, test_loader)
    print('METRICS ON TEST DATA:')
    for i in metrics_dct:
        print(i, metrics_dct[i])

    np.savetxt('predicted_labels.csv', np.array(predicted_lst), delimiter=',')

    