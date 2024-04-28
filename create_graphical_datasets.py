import argparse
import os

import pandas as pd
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and resize them to a specified format and size.")
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Directory containing folders of images categorized by label.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the processed images will be saved.')
    parser.add_argument('--image_format', type=str, choices=['png', 'jpg'], required=True,
                        help='Output image format (png or jpg).')
    parser.add_argument('--image_size', type=int, required=True,
                        help='Size of the output image, specified as one integer for both height and width.')
    parser.add_argument('--grayscale', action='store_true',
                        help='Convert images to grayscale. Omit this flag for RGB processing.')
    return parser.parse_args()


def create_graphical_datasets(source_dir, output_dir, image_format, image_size, grayscale=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = os.listdir(source_dir)
    data = []

    for category in categories:
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue

        print(f"Processing category: {category}")

        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        images = os.listdir(category_path)
        processed_count = 0
        skipped_count = 0

        for i, image_name in enumerate(images):
            processed_count += 1
            image_path = os.path.join(category_path, image_name)
            try:
                with Image.open(image_path) as img:
                    if grayscale:
                        img = img.convert('L')
                    else:
                        img = img.convert('RGB')
                    img = img.resize((image_size, image_size), Image.LANCZOS)
                    new_image_path = os.path.join(output_category_path, f"{processed_count:03d}.{image_format}")
                    img.save(new_image_path)
            except Exception as e:
                print(f"  Skipping {image_path}: {e}")
                skipped_count += 1
                processed_count -= 1

        data.append([category, processed_count, skipped_count])

    df = pd.DataFrame(data, columns=['Category', 'Processed', 'Skipped'])
    df.to_csv(os.path.join(output_dir, 'report.csv'), index=False)
    print(f"Processing complete. Report saved to {output_dir}/report.csv.")


if __name__ == "__main__":
    args = parse_arguments()
    create_graphical_datasets(args.source_dir, args.output_dir, args.image_format, args.image_size, args.grayscale)
