import os
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images in a folder.')
    parser.add_argument('path_to_images', type=str, help='Path to the folder containing images.')
    parser.add_argument('--output_file', type=str, default='captions.jsonl', help='Output file to write captions.')
    args = parser.parse_args()

    path_to_images = args.path_to_images
    output_file = args.output_file

    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True
    )
    model.to("cuda:0")

    # Instruction prompt
    instruction_prompt = """
        Describe this image in two parts. First, provide a single, brief phrase (up to 3 words) summarizing the overall picture.
        Then, describe the specific parts in detail, focusing solely on colors, shapes, and main elements. Be precise and direct, without additional commentary.
        Return the response in the following JSON structure: {'Overall picture': '<>', 'Specific details': '<>'}.
    """

    def generate_caption(image):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction_prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # Autoregressively complete the prompt
        output = model.generate(**inputs, max_new_tokens=128)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption

    # Get a sorted list of image files
    image_files = sorted(Path(path_to_images).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    # Avoid duplicates
    processed_filenames = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_filenames.add(data['filename'])
                except json.JSONDecodeError:
                    continue

    # Open the output file in append mode
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for image_path in tqdm(image_files, desc='Processing images'):
            filename = image_path.name
            if filename in processed_filenames:
                continue  # Skip already processed images
            try:
                image = Image.open(image_path)

                # Convert image to RGB
                # It prevents transparent background misrecognition
                if image.mode == 'RGBA':
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                else:
                    image = image.convert('RGB')

                caption = generate_caption(image)
                # Write the caption to the output file
                json_line = json.dumps({'filename': filename, 'caption': caption}, ensure_ascii=False)
                f_out.write(json_line + '\n')
            except Exception as e:
                print(f'Failed to generate caption for image {filename}: {e}')
                continue

if __name__ == '__main__':
    main()

"""
Example usage

python3 generate_captions.py path/to/png/folder --output_file captions.json

!!! It requires at least 8 GB of VRAM
"""

