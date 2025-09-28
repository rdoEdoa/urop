import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import Identity

# Brevitas imports
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8ActPerTensorFixedPoint, Uint8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint

import os

# Specify the directory for saving checkpoints
checkpoint_dir = './acc_2b/checkpoints_4b'
os.makedirs(checkpoint_dir, exist_ok=True)

# Variable parameters definition
bit_width = 4
num_epochs = 40
desired_epoch = 40
acc_bit = 2

# Save model and optimizer state
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir, filename="checkpoint.pth.tar"):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def print_accumulator_bitwidth(model):
    print("\n=== Accumulator Bitwidth Summary ===")
    
    # Direct approach: Extract the bit_width parameter from model definition
    # This is more reliable than trying to infer it from various attributes
    input_bit_width = None
    
    # Look for the QuantIdentity instance that's used as the input quantizer
    if hasattr(model, 'quant_input'):
        quant_input = model.quant_input
        if hasattr(quant_input, 'bit_width_param'):
            # Extract the actual parameter value
            try:
                input_bit_width = quant_input.bit_width_param.data.item()
                print(f"✅ Found input bitwidth from model.quant_input parameter: {input_bit_width} bits")
            except:
                pass
        elif hasattr(quant_input, '_bit_width'):
            try:
                input_bit_width = quant_input._bit_width
                print(f"✅ Found input bitwidth from model.quant_input._bit_width: {input_bit_width} bits")
            except:
                pass
        # Try calling bit_width() function if available
        elif hasattr(quant_input, 'bit_width') and callable(quant_input.bit_width):
            try:
                input_bit_width = quant_input.bit_width().item()
                print(f"✅ Found input bitwidth from model.quant_input.bit_width(): {input_bit_width} bits")
            except:
                pass
    
    # If we couldn't get it directly, check the model definition code
    if input_bit_width is None:
        # Inspect the model class to find bit_width attribute
        if hasattr(model.__class__, '__init__'):
            import inspect
            init_code = inspect.getsource(model.__class__.__init__)
            # Look for 'bit_width = ' in the initialization code
            import re
            match = re.search(r'bit_width\s*=\s*(\d+)', init_code)
            if match:
                input_bit_width = int(match.group(1))
                print(f"✅ Found input bitwidth from model class definition: {input_bit_width} bits")
    
    # Final fallback based on the naming convention used
    if input_bit_width is None and "Int8" in str(model.__class__):
        input_bit_width = 8
        print(f"✅ Inferred input bitwidth from Int8 in class names: {input_bit_width} bits")
    
    # Last resort fallback
    if input_bit_width is None:
        input_bit_width = 8
        print(f"⚠️ Could not find input bitwidth in model, using {input_bit_width} as default based on common practice")
        print(f"   Note: This may not be accurate - verify with your model configuration")
    
    # Now process each quantized layer
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            try:
                # Extract weight bitwidth - this is typically more reliable to extract
                weight_bit_width = None
                
                # Method 1: Try to get from weight_quant
                if hasattr(module, 'weight_quant') and hasattr(module.weight_quant, 'bit_width') and callable(module.weight_quant.bit_width):
                    try:
                        weight_bit_width = module.weight_quant.bit_width().item()
                    except:
                        pass
                
                # Method 2: Try to get from weight_bit_width_param
                if weight_bit_width is None and hasattr(module, 'weight_bit_width_param'):
                    try:
                        weight_bit_width = module.weight_bit_width_param.data.item()
                    except:
                        pass
                
                # Fallback for weight
                if weight_bit_width is None:
                    # Check class definition for weight quantization
                    if "Int8Weight" in str(module.__class__) or "Int8Weight" in str(type(module)):
                        weight_bit_width = 8
                    else:
                        # Default to the same as input
                        weight_bit_width = input_bit_width
                        
                # Calculate the accumulator bitwidth based on the layer type
                if isinstance(module, QuantConv2d):
                    # For convolutional layers
                    kernel_h, kernel_w = module.kernel_size
                    in_channels = module.in_channels
                    
                    # Log2 of MACs per output
                    n_macs = kernel_h * kernel_w * in_channels
                    log2_macs = int(np.ceil(np.log2(n_macs)))
                    
                    # The accumulator needs: input_bits + weight_bits + log2(MACs)
                    acc_bit_width = input_bit_width + weight_bit_width + log2_macs
                    
                    print(f"🔹 {name}: Accumulator bitwidth = {acc_bit_width} bits")
                    print(f"   └── Details: input={input_bit_width} + weight={weight_bit_width} + log2(MACs={n_macs})={log2_macs}")
                
                elif isinstance(module, QuantLinear):
                    # For linear layers
                    in_features = module.in_features
                    log2_features = int(np.ceil(np.log2(in_features)))
                    
                    # The accumulator needs: input_bits + weight_bits + log2(in_features)
                    acc_bit_width = input_bit_width + weight_bit_width + log2_features
                    
                    print(f"🔹 {name}: Accumulator bitwidth = {acc_bit_width} bits")
                    print(f"   └── Details: input={input_bit_width} + weight={weight_bit_width} + log2(features={in_features})={log2_features}")
            
            except Exception as e:
                print(f"⚠️ {name}: Error calculating accumulator bitwidth: {str(e)}")
    
    print("\n📌 Note on accuracy of the calculation:")
    print("   1. The accumulator bitwidth is calculated as: input_bits + weight_bits + log2(operations)")
    print("   2. For inference, this is the theoretical minimum bitwidth needed to avoid overflow")
    print("   3. The actual implemented bitwidth may be higher (often rounded to 16, 24, or 32 bits)")
    print("   4. To verify the exact accumulator bitwidth, check the hardware implementation details")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters

batch_size = 4
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Quantized CNN
class QuantCNN(torch.nn.Module):
    def __init__(self):
        super(QuantCNN, self).__init__()
        
        self.quant_input = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, bit_width=bit_width, return_quant_tensor=True)
        self.relu = QuantReLU(act_quant=Uint8ActPerTensorFixedPoint, bit_width=bit_width, return_quant_tensor=True)

        self.conv1 = QuantConv2d(3, 32, 3, padding=1, weight_quant=Int8WeightPerTensorFixedPoint, return_quant_tensor=True, accumulator_bit_width=acc_bit)
        self.conv2 = QuantConv2d(32, 64, 3, padding=1, weight_quant=Int8WeightPerTensorFixedPoint, return_quant_tensor=True, accumulator_bit_width=acc_bit)
        self.conv3 = QuantConv2d(64, 128, 3, padding=1, weight_quant=Int8WeightPerTensorFixedPoint, return_quant_tensor=True, accumulator_bit_width=acc_bit)
        self.conv4 = QuantConv2d(128, 256, 3, padding=1, weight_quant=Int8WeightPerTensorFixedPoint, return_quant_tensor=True, accumulator_bit_width=acc_bit)

        self.pool = torch.nn.MaxPool2d(2, 2)  # restore this if it was replaced
        self.dropout = torch.nn.Dropout(0.5)

        self._get_flatten_size()

        self.fc1 = QuantLinear(self.flatten_size, 256, bias=True, weight_quant=Int8WeightPerTensorFixedPoint, return_quant_tensor=True, accumulator_bit_width=acc_bit)

        self.fc2 = QuantLinear(256, 128, bias=True, weight_quant=Int8WeightPerTensorFixedPoint, return_quant_tensor=True, accumulator_bit_width=acc_bit)

        self.fc3 = QuantLinear(128, 10, bias=True, weight_quant=Int8WeightPerTensorFixedPoint, return_quant_tensor=False, accumulator_bit_width=acc_bit)

    def _get_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)  # shape must match input
            dummy = self.quant_input(dummy)   # <- add this line
            dummy = self.pool(self.relu(self.conv1(dummy)))
            dummy = self.pool(self.relu(self.conv2(dummy)))
            dummy = self.pool(self.relu(self.conv3(dummy)))
            dummy = self.pool(self.relu(self.conv4(dummy)))
            self.flatten_size = dummy.shape[1] * dummy.shape[2] * dummy.shape[3]
            self.flatten_size = 1024



    def forward(self, x):
        x = self.quant_input(x)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Create and train the model
model = QuantCNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# === Try to load existing checkpoint ===
import re

# Find all checkpoints in the directory
checkpoints = {}
for f in os.listdir(checkpoint_dir):
    match = re.match(r'checkpoint_epoch_(\d+)\.pth\.tar', f)
    if match:
        epoch_num = int(match.group(1))
        checkpoints[epoch_num] = os.path.join(checkpoint_dir, f)

# Load the specific checkpoint if it exists
if desired_epoch in checkpoints:
    checkpoint_path = checkpoints[desired_epoch]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"✅ Loaded checkpoint from epoch {desired_epoch} ({checkpoint_path}), resuming from epoch {start_epoch}")
else:
    model.to(device)
    start_epoch = 0
    print(f"🚫 No checkpoint found for epoch {desired_epoch}, starting from scratch.")


n_total_steps = len(train_loader)
for epoch in range(start_epoch, num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    save_checkpoint(epoch + 1, model, optimizer, loss.item(), checkpoint_dir, filename=f'checkpoint_epoch_{epoch+1}.pth.tar')

print('Finished Training')

# Evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.2f} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.2f} %')

# with torch.no_grad():
#     dummy_input = torch.randn(1, 3, 32, 32).to(device)
#     model(dummy_input)
print_accumulator_bitwidth(model)

     