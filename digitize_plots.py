import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Digitize plots from images")
    parser.add_argument("--upscale", action="store_true", help="Upscale images")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to file containing image paths/URLs",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=12800, help="Max number of tokens to generate"
    )
    parser.add_argument(
        "--use_flash_attention", action="store_true", help="Use flash attention"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Extract the underlying data for this plot in json format.",
        help="Prompt to use for digitizing plots",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for processing images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.json",
        help="Path to save the output JSON file",
    )
    return parser.parse_args()


def process_image_batch(image_paths, processor, model, max_new_tokens, prompt):
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image_path in image_paths
    ]

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_texts


def main():
    args = parse_arguments()

    # Set up model and processor
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

    processor_kwargs = {}
    if args.upscale:
        processor_kwargs = {
            "min_pixels": 1000 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }

    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    # Process images
    with open(args.input_file, "r") as f:
        image_paths = f.read().splitlines()

    results = {}
    for i in range(0, len(image_paths), args.batch_size):
        batch = image_paths[i : i + args.batch_size]
        outputs = process_image_batch(
            batch, processor, model, args.max_tokens, args.prompt
        )
        for image_path, output in zip(batch, outputs):
            results[image_path] = output

    # Save results to file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
