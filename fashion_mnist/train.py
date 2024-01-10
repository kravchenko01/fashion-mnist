# import matplotlib.pyplot as plt
import hydra
import torch
from model import FashionCNN
from omegaconf import DictConfig  # , OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def train_model(model, train_loader, num_epoch, lr, device):
    model.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    count = 0
    loss_list = []
    iteration_list = []
    for _ in range(num_epoch):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

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


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Загружаем данные, обучаем модель и сохраняем на диск
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = datasets.FashionMNIST(
        root=cfg.data.path,
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True)

    # image, label = next(iter(train_set))
    # print(label, image.shape)
    # plt.imshow(image.squeeze(), cmap="gray")
    # plt.savefig("./example.png")

    model = FashionCNN(num_classes=cfg.model.num_classes)
    model.to(device)

    loss_list, iteration_list = train_model(
        model,
        train_loader,
        num_epoch=cfg.train.epochs,
        lr=cfg.train.learning_rate,
        device=device,
    )

    torch.save(model.state_dict(), "./trained_weights_FashionCNN.pth")

    # plt.plot(iteration_list, loss_list)
    # plt.xlabel("No. of Iteration")
    # plt.ylabel("Loss")
    # plt.title("Iterations vs Loss")
    # plt.savefig("./training_loss.png")


if __name__ == "__main__":
    main()
