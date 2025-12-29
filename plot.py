import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif"})

    plt.figure(figsize=(8, 6))
    with open("error_train.csv", "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        model_names = header[1:]

        for idx, model_name in enumerate(model_names):
            error_train = [
                float(line.strip().split(",")[idx + 1]) for line in lines[1:]
            ]
            print(
                f"Last iteration average training error for {model_name}: {np.mean(error_train[-391:]):.2f}%"
            )

            plt.plot(
                np.arange(1, len(error_train) + 1),
                error_train,
                label=model_name.replace("vgg", "VGG-").replace("resnet", "ResNet-"),
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
        lines = f.readlines()
        header = lines[0].strip().split(",")
        model_names = header[1:]

        for idx, model_name in enumerate(model_names):
            error_test = [float(line.strip().split(",")[idx + 1]) for line in lines[1:]]
            print(f"The lowest test error for {model_name}: {min(error_test):.2f}%")

            plt.plot(
                np.arange(1, len(error_test) + 1),
                error_test,
                label=model_name.replace("vgg", "VGG-").replace("resnet", "ResNet-"),
                marker="o",
            )

        epochs = len(lines)
        plt.title("CIFAR-10 Test Error Rate per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Error (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(np.arange(1, epochs + 1))
        plt.legend()
        plt.savefig("error_test_plot.pdf", format="pdf", bbox_inches="tight")
        print("Saved test error plot as 'error_test_plot.pdf'.")
