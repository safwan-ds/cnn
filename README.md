# CNN Architectures on CIFAR-10

A PyTorch implementation of classic CNN architectures (VGG and ResNet) for image classification on the CIFAR-10 dataset.

## Features

- **VGG Networks**: VGG-11, VGG-13, VGG-16, VGG-16 (strided), and VGG-19
- **ResNet**: ResNet-18, ResNet-34, ResNet-50, ResNet-101
- **Plain Networks**: Plain-18 (ResNet without skip connections for comparison)
- Training with data augmentation (random crop, horizontal flip)
- Cosine annealing learning rate scheduler
- Automatic logging of training and test error rates
- Plotting utilities for visualizing results

## Project Structure

```text
cnn/
├── main.py           # Training script
├── vgg.py            # VGG network implementations
├── resnet.py         # ResNet and plain network implementations
├── plot.py           # Plotting utilities for error rates
├── data/             # CIFAR-10 dataset (auto-downloaded)
├── results/          # Training results (CSV files)
└── plots/            # Generated plots
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Usage

### Training

Edit the `USED_MODELS` dictionary in `main.py` to select which models to train:

```python
USED_MODELS = {
    vgg16: True,       # Set to True to train
    resnet18: True,
    plain18: True,
    # ... other models
}
```

Run training:

```bash
python main.py
```

Training configuration:

- **Epochs**: 30 (default)
- **Batch size**: 128
- **Optimizer**: SGD with momentum (0.9) and weight decay (5e-4)
- **Initial learning rate**: 0.1
- **LR scheduler**: Cosine annealing

### Plotting Results

After training, generate error plots:

```bash
python plot.py
```

This will create:

- `error_train_plot.pdf` — Training error per iteration
- `error_test_plot.pdf` — Test error per epoch

## Models

### VGG Networks

| Model            | Conv Layers | Parameters | Description                                |
| ---------------- | ----------- | ---------- | ------------------------------------------ |
| VGG-11           | 8           | ~133M      | Smaller VGG variant                        |
| VGG-13           | 10          | ~133M      | Two conv layers per block                  |
| VGG-16           | 13          | ~138M      | Original VGG-16 with max pooling           |
| VGG-16 (strided) | 13          | ~138M      | Pooling replaced with strided convolutions |
| VGG-19           | 16          | ~144M      | Deepest VGG variant                        |

### ResNet

| Model      | Layers | Block Type | Description                        |
| ---------- | ------ | ---------- | ---------------------------------- |
| ResNet-18  | 18     | BasicBlock | Lightweight ResNet                 |
| ResNet-34  | 34     | BasicBlock | Deeper with basic blocks           |
| ResNet-50  | 50     | Bottleneck | Uses bottleneck blocks             |
| ResNet-101 | 101    | Bottleneck | Very deep ResNet                   |
| Plain-18   | 18     | BasicBlock | ResNet-18 without skip connections |

## Results

Training and test error rates are saved to the `results/` directory:

- `error_train_<model>.csv` — Per-iteration training error
- `error_test_<model>.csv` — Per-epoch test error

## References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (VGG)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
- [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806) (Strided convolutions)

## License

MIT License
