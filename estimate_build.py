import argparse
import os
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow_steps import *


def main():
    # Configura il parser degli argomenti
    parser = argparse.ArgumentParser(
        description='Execute the FINN build process on a specified ONNX model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('directory', 
                        help='Directory that contains the ONNX model and where the build output will be saved, located inside models/')
    parser.add_argument('--model-name', '-m',
                        default='model.onnx',
                        help='Name of the ONNX model file located in the specified directory')
    parser.add_argument('--output-dir', '-o',
                        default='build',
                        help='Name of the output directory for the build results inside the model directory')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Detailed output log')
    parser.add_argument('--folding-config', '-fc',
                        default='folding_config.json',
                        help='Set the folding configuration file name located in the config directory')

    # Parse of the arguments
    args = parser.parse_args()

    # Build paths
    model_directory = os.path.join("models", args.directory)
    if not os.path.exists(model_directory):
        print(f"Error: The model directory '{model_directory}' does not exist")
        return 1

    model_file = os.path.join("models", args.directory, args.model_name)
    # Check if the model file exists
    if not os.path.exists(model_file):
        print(f"Error: The model '{model_file}' does not exist")
        return 1

    # If the folding config file is not the default, append its name to the output directory
    if args.folding_config != "folding_config.json":
        output_dir = os.path.join("models", args.directory, args.output_dir + "_" + os.path.splitext(args.folding_config)[0])
    else:
        output_dir = os.path.join("models", args.directory, args.output_dir)

    folding_config_path = os.path.join("config", args.folding_config)

    if args.verbose:
        print("-" * 50)
        print(f"Working directory: {model_directory}")
        print(f"Model: {model_file}")
        print(f"Output directory: {output_dir}")
        print(f"Folding configuration file: {folding_config_path}")
        print("-" * 50)


    model = ModelWrapper(model_file)

    cfg = DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        hls_clk_period_ns=10.0,
        generate_outputs={DataflowOutputType.ESTIMATE_REPORTS},
        folding_config_file=folding_config_path,
        fpga_part="xc7z020clg400-1"
    )
    cfg.output_types = ["estimate"]
    cfg.mvau_optimization = "resource"

    estimate_only_dataflow_steps = [
        "step_qonnx_to_finn",
        "step_tidy_up",
        "step_streamline",
        "step_convert_to_hw",
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
    ]

    step_lookup = build_dataflow_step_lookup  
    save_model = True

    for i, step_name in enumerate(estimate_only_dataflow_steps):
        print(f"Running step: {step_name} [{i+1}/{len(estimate_only_dataflow_steps)}]")
        step_function = step_lookup[step_name]
        model = step_function(model, cfg)

        if step_name == "step_convert_to_hw":
            print("\n=== Node names after hardware conversion ===")
            for node in model.graph.node:
                print(node.name)
            print("============================================\n")
    return 0

if __name__ == "__main__":
    main()
