import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif"})

    # Find all training error files
    train_files = glob.glob("results/error_train_*.csv")

    if train_files:
        plt.figure(figsize=(8, 6))
        for filepath in train_files:
            model_name = (
                os.path.basename(filepath)
                .replace("error_train_", "")
                .replace(".csv", "")
            )
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                error_train = [float(row[1]) for row in reader]

            print(
                f"Last batch average training error for {model_name}: {np.mean(error_train[-391:]):.2f}%"
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
        plt.savefig("plots/error_train_plot.pdf", format="pdf", bbox_inches="tight")
        print("Saved training error plot as 'plots/error_train_plot.pdf'.")

    # Find all test error files
    test_files = glob.glob("results/error_test_*.csv")

    if test_files:
        plt.figure(figsize=(8, 6))
        epochs = 0
        for filepath in test_files:
            model_name = (
                os.path.basename(filepath)
                .replace("error_test_", "")
                .replace(".csv", "")
            )
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                error_test = [float(row[1]) for row in reader]

            print(f"The lowest test error for {model_name}: {min(error_test):.2f}%")
            epochs = max(epochs, len(error_test))

            plt.plot(
                np.arange(1, len(error_test) + 1),
                error_test,
                label=model_name.replace("vgg", "VGG-")
                .replace("resnet", "ResNet-")
                .replace("plain", "plain-"),
                marker="o",
            )

        plt.title("CIFAR-10 Test Error Rate per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Error (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(np.arange(1, epochs + 1))
        plt.legend()
        plt.savefig("plots/error_test_plot.pdf", format="pdf", bbox_inches="tight")
        print("Saved test error plot as 'plots/error_test_plot.pdf'.")
