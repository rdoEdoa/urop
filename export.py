import torch
import sys
from qat_cnn import QuantCNN  # Your quantized model
from brevitas.export import export_qonnx
import os

# Config
CHECKPOINT_PATH = "./acc_2b/checkpoints_4b/checkpoint_epoch_40.pth.tar"
ONNX_EXPORT_PATH = "quantcnn_acc2b_4b.onnx"
INPUT_SHAPE = (1, 3, 32, 32)

# Load model and weights
model = QuantCNN()
model.eval()

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("❌ Error: 'model_state_dict' missing in checkpoint")
    sys.exit(1)

# Dummy input to trace shape
dummy_input = torch.randn(*INPUT_SHAPE)

# Export using Brevitas (preserves quantization)
export_qonnx(model, dummy_input, export_path=ONNX_EXPORT_PATH, opset_version=9)

print(f"✅ Exported quantized model to ONNX: {ONNX_EXPORT_PATH}")