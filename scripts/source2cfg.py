import os
import json
import subprocess
from glob import glob
from collections import defaultdict

# Base directory containing subfolders with Solidity contracts
BASE_DIR = "dataset/cgt/source"
# Base output directory for CFGs
OUTPUT_BASE_DIR = "dataset/cgt/cfg"


def generate_function_cfgs(sol_file):
    """Generate function-level CFGs using Slither."""
    try:
        subprocess.run(["slither", sol_file, "--print", "cfg"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating CFG for {sol_file}: {e}")
        return False
    return True


def parse_dot_file(dot_file):
    """Parse a DOT file and return its nodes and edges."""
    nodes = set()
    edges = []

    with open(dot_file, "r") as f:
        for line in f:
            if "->" in line:  # Edge
                parts = line.split("->")
                src = parts[0].strip().strip('"')
                dst = parts[1].strip().strip('";')
                edges.append((src, dst))
                nodes.add(src)
                nodes.add(dst)
            elif "[" in line:  # Node
                node = line.split("[")[0].strip().strip('"')
                nodes.add(node)

    return nodes, edges


def combine_cfgs(sol_file):
    """Combine individual function-level CFGs into a single graph."""
    nodes = set()
    edges = []

    for dotfile in glob(f"{sol_file}-*.dot"):
        fn_nodes, fn_edges = parse_dot_file(dotfile)
        nodes.update(fn_nodes)
        edges.extend(fn_edges)

    return {"nodes": list(nodes), "edges": [{"source": src, "target": dst} for src, dst in edges]}


def save_json(cfg, output_file):
    """Save the CFG to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(cfg, f, indent=2)


def generate_combined_cfg(sol_file, output_dir):
    """Generate and save the combined CFG for a Solidity file."""
    # Prepare output filename
    filename = os.path.basename(sol_file)
    contract_name = os.path.splitext(filename)[0]
    output_json = os.path.join(output_dir, f"{contract_name}-combined.json")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate function-level CFGs
    print(f"Generating function-level CFGs for {sol_file}...")
    if not generate_function_cfgs(sol_file):
        print(f"Skipping {sol_file} due to compilation errors.")
        return

    # Combine function-level CFGs
    print(f"Combining CFGs into one graph for {contract_name}...")
    combined_cfg = combine_cfgs(sol_file)

    # Save the combined CFG as JSON
    print(f"Saving combined CFG to {output_json}...")
    save_json(combined_cfg, output_json)

    # Clean up individual .dot files
    for dotfile in glob(f"{sol_file}-*.dot"):
        os.remove(dotfile)

    print(f"Combined CFG generated and saved to {output_json}")


def process_directory(base_dir, output_base):
    """Process all Solidity files in the directory recursively."""
    for root, _, files in os.walk(base_dir):
        output_dir = os.path.join(output_base, f"cfg_{os.path.basename(root)}")
        for file in files:
            if file.endswith(".sol"):
                sol_file = os.path.join(root, file)
                print(f"Processing file: {sol_file}")
                generate_combined_cfg(sol_file, output_dir)


if __name__ == "__main__":
    process_directory(BASE_DIR, OUTPUT_BASE_DIR)
    print(f"All contracts processed. Combined CFGs are stored in {OUTPUT_BASE_DIR}")
