# CNN Architectures on CIFAR-10

PyTorch implementations of VGG and ResNet variants for CIFAR-10. Training logs errors to CSV and generates PDF plots for training, test, and generalization gap curves.

## What's inside

- VGG family: VGG-11/13/16/19 plus a strided VGG-16 without pooling (see [vgg.py](vgg.py)).
- ResNet family: ResNet-18/34/50/101 and a Plain-18 baseline without skip connections (see [resnet.py](resnet.py)).
- CIFAR-10 dataloaders with crop + flip augmentation and normalization; data auto-downloads to `./data` (see [main.py](main.py#L16-L44)).
- SGD with momentum, weight decay, and cosine annealing LR scheduler (see [main.py](main.py#L47-L90)).
- CSV logging per model and automatic plot generation for train error, test error, and generalization gap (see [plot.py](plot.py#L8-L143)).

## Project layout

```text
cnn/
├── main.py           # Training + logging entrypoint
├── vgg.py            # VGG network definitions (with strided/no-pooling variant)
├── resnet.py         # ResNet and plain network definitions
├── plot.py           # Plotting utilities for CSV logs
├── data/             # CIFAR-10 dataset cache (auto-downloaded)
├── results/          # CSV logs written per model
├── plots/            # Generated PDF plots
└── LICENSE
```

## Setup

- Python 3.8+
- Install deps: `pip install torch torchvision numpy matplotlib tqdm`

## Train

1. Choose models by toggling `USED_MODELS` in [main.py](main.py#L114-L125).
2. Optional: adjust `EPOCHS` and `BATCH_SIZE` in [main.py](main.py#L112-L113).
3. Run `python main.py` (downloads CIFAR-10 to `./data`, writes CSVs under `results/`, and saves plots to `plots/`).

Default training config: 30 epochs, batch size 128, SGD (lr 0.1, momentum 0.9, weight decay 5e-4), cosine annealing scheduler.

## Plot only

If CSV logs already exist, regenerate plots with `python plot.py`. This produces `plots/error_train_plot.pdf`, `plots/error_test_plot.pdf`, and `plots/generalization_gap_plot.pdf` (per-model curves are labeled automatically).

## Outputs

- Per model: `results/error_train_<model>.csv` (per-iteration train error) and `results/error_test_<model>.csv` (per-epoch test error).
- PDFs in `plots/`: training error, test error, and generalization gap across epochs.

## Models

### VGG

| Model            | Conv Layers | Parameters | Notes                                    |
| ---------------- | ----------- | ---------- | ---------------------------------------- |
| VGG-11           | 8           | ~133M      | Compact variant                          |
| VGG-13           | 10          | ~133M      | Two conv layers per block                |
| VGG-16           | 13          | ~138M      | Classic VGG-16 with max pooling          |
| VGG-16 (strided) | 13          | ~138M      | Pooling replaced by strided convolutions |
| VGG-19           | 16          | ~144M      | Deepest VGG variant                      |

### ResNet

| Model      | Layers | Block Type | Notes                              |
| ---------- | ------ | ---------- | ---------------------------------- |
| ResNet-18  | 18     | BasicBlock | Lightweight ResNet                 |
| ResNet-34  | 34     | BasicBlock | Deeper basic-block stack           |
| ResNet-50  | 50     | Bottleneck | Bottleneck blocks                  |
| ResNet-101 | 101    | Bottleneck | Very deep bottleneck configuration |
| Plain-18   | 18     | BasicBlock | ResNet-18 without skip connections |

## References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)

## License

MIT License — see [LICENSE](LICENSE).
