import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet import resnet18, plain18, resnet34, resnet50, resnet101
from vgg import vgg11, vgg13, vgg16, vgg19


def get_data_loaders(batch_size=128):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return trainloader, testloader


def train(model, trainloader, testloader, device, epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    error_train = []
    error_test = []

    model.train()
    print(f"Starting training {model.__class__.__name__}...")

    for epoch in range(epochs):
        correct = 0
        total = 0
        pbar = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for _, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / total

            pbar.set_postfix({"error": f"{100 - acc:.2f}%"})

            error_train.append(100 - acc)

        scheduler.step()

        current_error = (1 - test(model, testloader, device)) * 100
        error_test.append(current_error)

    return error_train, error_test


def test(model: nn.Module, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f" -> Test Accuracy: {100 * acc:.2f}% (Error: {100 * (1 - acc):.2f}%)")
    model.train()
    return acc


if __name__ == "__main__":
    EPOCHS = 30
    BATCH_SIZE = 128
    USED_MODELS = {
        vgg11: False,
        vgg13: False,
        vgg16: True,
        vgg19: False,
        resnet18: True,
        plain18: True,
        resnet34: False,
        resnet50: False,
        resnet101: False,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_data_loaders(batch_size=BATCH_SIZE)
    models = [model(num_classes=10) for model, used in USED_MODELS.items() if used]
    results = []

    for model in models:
        model.to(device)
        results.append(train(model, trainloader, testloader, device, epochs=EPOCHS))

    with open("error_train.csv", "w") as f:
        f.write(
            f"iteration,{','.join([model.__name__ for model, used in USED_MODELS.items() if used])}\n"
        )
        for iter in range(len(results[0][0])):
            f.write(
                f"{iter+1},{','.join([f'{results[j][0][iter]:.6f}' for j in range(len(results))])}\n"
            )

    with open("error_test.csv", "w") as f:
        f.write(
            f"epoch,{','.join([model.__name__ for model, used in USED_MODELS.items() if used])}\n"
        )
        for i in range(EPOCHS):
            f.write(
                f"{i+1},{','.join([f'{results[j][1][i]:.2f}' for j in range(len(results))])}\n"
            )

    print("Training complete. Results saved to 'error_train.csv' and 'error_test.csv'.")
