import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif"})

    if not os.path.exists("plots"):
        os.makedirs("plots")

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
                .replace("plain", "plain-")
                .replace("_no_pooling", " (no pooling)"),
                linewidth=0.75,
            )

        plt.title("CIFAR-10 Training Error per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Error (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.savefig("plots/error_train_plot.pdf", format="pdf", bbox_inches="tight")
        print("Saved training error plot as 'plots/error_train_plot.pdf'.")

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
                .replace("plain", "plain-")
                .replace("_no_pooling", " (no pooling)"),
                marker="o",
            )

        plt.title("CIFAR-10 Test Error Rate per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Error (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(np.arange(0, epochs + 1, 2))
        plt.legend()
        plt.savefig("plots/error_test_plot.pdf", format="pdf", bbox_inches="tight")
        print("Saved test error plot as 'plots/error_test_plot.pdf'.")

    if train_files and test_files:
        plt.figure(figsize=(8, 6))
        max_epochs = 0

        for filepath in test_files:
            model_name = (
                os.path.basename(filepath)
                .replace("error_test_", "")
                .replace(".csv", "")
            )

            test_path = f"results/error_test_{model_name}.csv"
            with open(test_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                error_test = [float(row[1]) for row in reader]

            train_path = f"results/error_train_{model_name}.csv"
            with open(train_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                error_train = [float(row[1]) for row in reader]

            n_epochs = len(error_test)
            max_epochs = max(max_epochs, n_epochs)
            iters_per_epoch = len(error_train) // n_epochs
            error_train_per_epoch = []
            for i in range(n_epochs):
                start_idx = i * iters_per_epoch
                end_idx = (i + 1) * iters_per_epoch
                epoch_avg = np.mean(error_train[start_idx:end_idx])
                error_train_per_epoch.append(epoch_avg)

            gap = np.array(error_test) - np.array(error_train_per_epoch)
            epochs_arr = np.arange(1, n_epochs + 1)

            display_name = (
                model_name.replace("vgg", "VGG-")
                .replace("resnet", "ResNet-")
                .replace("plain", "plain-")
                .replace("_no_pooling", " (no pooling)")
            )

            plt.plot(epochs_arr, gap, label=display_name, marker="o")

        plt.title("CIFAR-10 Generalization Gap (Test Error - Training Error)")
        plt.xlabel("Epoch")
        plt.ylabel("Generalization Gap (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(np.arange(0, max_epochs + 1, 2))
        plt.legend()
        plt.savefig(
            "plots/generalization_gap_plot.pdf", format="pdf", bbox_inches="tight"
        )
        print("Saved generalization gap plot as 'plots/generalization_gap_plot.pdf'.")
