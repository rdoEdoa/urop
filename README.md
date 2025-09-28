# Deployment of a quantized neural network on FPGA

First, to properly use the scripts for the first parts, it is necessary to install some packages. To do so, it is sufficient to run:

```
pip install -r requirements.txt
```

## Training

The file `qat_cnn.py` contains the model of the convolutional neural network, with quantization-aware training using the brevitas library. It is necessary to set the needed parameters at the beginning of the file, from line 21 to line 24:

- `bit_width`: is the bit-width of the network.
- `num_epochs`: is the total amount of epochs performed during training.
- `desired_epochs`: the checkpoint at the end of each epoch is saved; with this parameter it is possible to select from where to resume the training. If the corresponding checkpoint is not found, the training starts from scratch.
- `acc_bit`: is the accumulator bit-width for all the layers.

Moreover, it is necessary to set the directory in which the checkpoints are saved, and where the script will search for already existing checkpoints, in the parameter `checkpoint_dir` at line 17.

## Model export


