# Deployment of a quantized neural network on FPGA

First, to properly use the scripts for the first parts, it is necessary to install some packages. To do so, it is sufficient to run:

```
pip install -r requirements.txt
```

Note that some packages, like `onnxoptimizer` are not supported on recent version of the Python environment. It is suggested to use Python 3.11. Moreover, it is important to check and install the correct version of the packages based on the system used (it is not always compatible).

## Training

The file `qat_cnn.py` contains the model of the convolutional neural network, with quantization-aware training using the brevitas library. It is necessary to set the needed parameters at the beginning of the file, from line 21 to line 24:

- `bit_width`: is the bit-width of the network.
- `num_epochs`: is the total amount of epochs performed during training.
- `desired_epochs`: the checkpoint at the end of each epoch is saved; with this parameter it is possible to select from where to resume the training. If the corresponding checkpoint is not found, the training starts from scratch.
- `acc_bit`: is the accumulator bit-width for all the layers.

Moreover, it is necessary to set the directory in which the checkpoints are saved, and where the script will search for already existing checkpoints, in the parameter `checkpoint_dir` at line 17.

## Model export

For FINN, it is necessary to have the `.onnx` model file. To obtain it, run the corresponding script:

1) Set in `CHECKPOINT_PATH` the correct path to checkpoint to export, including its name.
2) Set in `ONNX_EXPORT_PATH` the path and the name of the model to export; it needs to have the correct file extension `.onnx`.
3) Run the script:

```
python export.py
```

## FINN build estimation
To use FINN, first it needs to be properly installed and configured. To do so, follow the Quickstart guide on the official website:

https://finn.readthedocs.io/en/latest/getting_started.html

Then, while inside the FINN environment (to enter it, just run `./run-docker` while inside the cloned repo), it is sufficient to run the build_finn.py script. This will prepare the model conversion to HW and produce a set of reports with the estimation of some parameters of the generated model (like timing, resource usage, etc.). 


