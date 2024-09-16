# Plot Digitization Script

## Requirements

Before running the script, ensure you have the following dependencies installed:

```bash
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
pip install flash-attn
pip install torch
```

## Usage

To run the script, use the following command:

```bash
python gpu_cluster_script.py --input_file <path_to_input_file> [options]
```

The `<path_to_input_file>` should be a text file containing one image path or URL per line.

## Options

The script supports the following command-line options:

- `--input_file` (required): Path to a file containing image paths/URLs (one per line)
- `--upscale`: Flag to enable image upscaling. This might be helpful for small/low-quality images
- `--max_tokens` (default: 12800): Maximum number of tokens to generate
- `--use_flash_attention`: Flag to enable flash attention. Use on more recent GPUs for speedup
- `--prompt` (default: "Extract the underlying data for this plot in json format."): Custom prompt for image analysis
- `--output_file` (default: "output.json"): Path to save the output JSON file

## Output

The script processes images in batches and saves the results to a JSON file. The output file contains a dictionary where:
- Keys are the image paths/URLs
- Values are the generated text based on the prompt for each image