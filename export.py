import os
import sys
import torch
import types
from qat_cnn import QuantCNN
from brevitas.export import export_qonnx
import brevitas.export.onnx.qonnx.manager as qmanager

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA globally

CHECKPOINT_PATH = "./checkpoints_8b/checkpoint_epoch_40.pth.tar"
ONNX_EXPORT_PATH = "model.onnx"
INPUT_SHAPE = (1, 3, 32, 32)

# Disable optimizer passes
qmanager.QONNXManager.onnx_passes = []

# --- Patch onnxoptimizer if needed ---
try:
    import onnxoptimizer
    if not hasattr(onnxoptimizer, "optimize"):
        print("⚠️  Patching onnxoptimizer: adding dummy optimize()")
        def _dummy_optimize(model, passes=None):
            print("⚠️  Skipping onnxoptimizer.optimize() (no-op patch)")
            return model
        onnxoptimizer.optimize = _dummy_optimize
except ImportError:
    print("⚠️  onnxoptimizer not installed; using dummy module")
    onnxoptimizer = types.SimpleNamespace(optimize=lambda model, passes=None: model)
    sys.modules["onnxoptimizer"] = onnxoptimizer

# --- Load model and weights ---
model = QuantCNN().cpu()
model.eval()

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("❌ Error: 'model_state_dict' missing in checkpoint")
    sys.exit(1)

dummy_input = torch.randn(*INPUT_SHAPE, device='cpu')

# --- Export to ONNX ---
export_qonnx(model, dummy_input, export_path=ONNX_EXPORT_PATH, opset_version=9)

print(f"✅ Exported quantized model to ONNX: {ONNX_EXPORT_PATH}")
