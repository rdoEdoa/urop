import numpy as np
import os
import shutil
import glob
import argparse
import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from enum import Enum
from typing import Any, List, Optional

from finn.transformation.fpgadataflow.vitis_build import VitisOptStrategy
from finn.util.basic import alveo_default_platform, alveo_part_map, pynq_part_map

from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow_steps import *

def get_build_steps():
    """Gives the list of build steps to be executed"""
    return [
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
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
        "step_synthesize_bitfile",
        "step_make_pynq_driver",
        "step_deployment_package",
    ]


def create_build_config(output_dir):
    """Crea e configura il DataflowBuildConfig"""
    cfg = DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=20.0,
        generate_outputs=[
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.RTLSIM_PERFORMANCE,
            DataflowOutputType.BITFILE,
            DataflowOutputType.STITCHED_IP
        ],
        folding_config_file="folding_config.json",
        auto_fifo_depths=False,
        fpga_part="xc7z020clg400-1",
        folding_two_pass_relaxation=False,
        mvau_wwidth_max=256,
        split_large_fifos=True,
        enable_build_pdb_debug=True,
        standalone_thresholds=True,
        shell_flow_type="vivado_zynq"
    )
    cfg.output_types = [
        DataflowOutputType.ESTIMATE_REPORTS,
        DataflowOutputType.RTLSIM_PERFORMANCE,
        DataflowOutputType.OOC_SYNTH,
        DataflowOutputType.BITFILE,
        DataflowOutputType.PYNQ_DRIVER,
        DataflowOutputType.DEPLOYMENT_PACKAGE,
        DataflowOutputType.STITCHED_IP
    ]
    cfg.mvau_optimization = "resource"
    return cfg

def execute_build_steps(model, cfg, build_steps, verbose=False):
    """Execute the build steps on the model with the given configuration"""
    step_lookup = build_dataflow_step_lookup.copy()  # Copy the lookup
    
    for i, step_name in enumerate(build_steps):
        if verbose:
            print(f"Running step: {step_name} [{i+1}/{len(build_steps)}]")
        else:
            print(f"Step {i+1}/{len(build_steps)}: {step_name}")
            
        try:
            step_function = step_lookup[step_name]
            model = step_function(model, cfg)

            if step_name == "step_convert_to_hw":
                for n in model.graph.node:
                    if "MVAU" in n.op_type:
                        W = model.get_initializer(n.input[1])  # tensore dei pesi
                        print(n.name, W.shape)

                
        except Exception as e:
            print(f"Error during the execution of the step '{step_name}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return model

def collect_all_reports(output_dir):
    """Copy all FINN reports (JSON, TXT, LOG, CSV) recursively into a central folder"""
    reports_dir = os.path.join(output_dir, "all_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Estensioni dei report da salvare
    exts = ["json", "txt", "log", "csv"]
    
    for ext in exts:
        pattern = os.path.join(output_dir, "**", f"*.{ext}")  # ricerca ricorsiva
        for f in glob.glob(pattern, recursive=True):
            try:
                shutil.copy(f, reports_dir)
            except Exception as e:
                print(f"Warning: could not copy {f}: {e}")
    
    print(f"\nAll FINN reports collected in: {reports_dir}")


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
                        help='Output directory for the build results within the specified directory')

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
    
    output_dir = os.path.join("models", args.directory, "build")
    folding_config_path = os.path.join("config", args.folding_config)
    
    if args.verbose:
        print("-" * 50)
        print(f"Working directory: {model_directory}")
        print(f"Model: {model_file}")
        print(f"Output directory: {output_dir}")
        print(f"Folding configuration file: {folding_config_path}")
        print("-" * 50)
    
    try:
        # Load the model
        print("Loading the model...")
        model = ModelWrapper(model_file)
        
        # Create build configuration
        print("Create build configuration...")
        cfg = create_build_config(output_dir)
        
        # Update folding config path and FPGA part
        cfg.folding_config_file = folding_config_path
        cfg.fpga_part = "xc7z020clg400-1"
        
        # Get build steps
        build_steps = get_build_steps()
        
        print(f"Starting build with {len(build_steps)} steps...")
        print("=" * 50)
        
        # Execute build steps
        result_model = execute_build_steps(model, cfg, build_steps, args.verbose)
        
        if result_model is not None:
            print("=" * 50)
            print("Build executed successfully!")
            print(f"Result saved in: {output_dir}")
        else:
            print("Build failed!")
            return 1
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    collect_all_reports(output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
