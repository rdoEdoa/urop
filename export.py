import torch
import onnx
from collections import OrderedDict
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup

# Import your model
from model import QuantCNN  # Adjust if needed

def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get the actual state_dict inside the checkpoint
    state_dict = checkpoint["model_state_dict"]

    # Remove 'module.' if trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_k] = v

    model = QuantCNN()
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def export_and_cleanup_qonnx(model, dummy_input, qonnx_path, cleaned_path):
    export_qonnx(model, dummy_input, export_path=qonnx_path)
    print(f"Exported raw QONNX to: {qonnx_path}")

    model_qonnx = onnx.load(qonnx_path)
    model_clean = cleanup(model_qonnx)
    onnx.save(model_clean, cleaned_path)
    print(f"Saved cleaned FINN-compatible QONNX to: {cleaned_path}")

def main():
    checkpoint_path = "model_best.pth.tar"
    raw_qonnx_path = "quantcnn_qonnx_raw.onnx"
    cleaned_qonnx_path = "quantcnn_finn.onnx"

    model = load_model_from_checkpoint(checkpoint_path)
    dummy_input = torch.randn(1, 3, 32, 32)

    export_and_cleanup_qonnx(model, dummy_input, raw_qonnx_path, cleaned_qonnx_path)

if __name__ == "__main__":
    main()
