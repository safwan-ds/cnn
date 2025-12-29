import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif"})

    plt.figure(figsize=(8, 6))
    with open("error_train.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        model_names = header[1:]

        data = list(reader)
        for idx, model_name in enumerate(model_names):
            error_train = [float(row[idx + 1]) for row in data]
            print(
                f"Last iteration average training error for {model_name}: {np.mean(error_train[-391:]):.2f}%"
            )

            plt.plot(
                np.arange(1, len(error_train) + 1),
                error_train,
                label=model_name.replace("vgg", "VGG-")
                .replace("resnet", "ResNet-")
                .replace("plain", "plain-"),
                linewidth=0.75,
            )

        plt.title("CIFAR-10 Training Error per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Error (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.savefig("error_train_plot.pdf", format="pdf", bbox_inches="tight")
        print("Saved training error plot as 'error_train_plot.pdf'.")

    plt.figure(figsize=(8, 6))
    with open("error_test.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        model_names = header[1:]

        data = list(reader)
        for idx, model_name in enumerate(model_names):
            error_test = [float(row[idx + 1]) for row in data]
            print(f"The lowest test error for {model_name}: {min(error_test):.2f}%")

            plt.plot(
                np.arange(1, len(error_test) + 1),
                error_test,
                label=model_name.replace("vgg", "VGG-")
                .replace("resnet", "ResNet-")
                .replace("plain", "plain-"),
                marker="o",
            )

        epochs = len(data) + 1
        plt.title("CIFAR-10 Test Error Rate per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Error (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(np.arange(1, epochs))
        plt.legend()
        plt.savefig("error_test_plot.pdf", format="pdf", bbox_inches="tight")
        print("Saved test error plot as 'error_test_plot.pdf'.")
