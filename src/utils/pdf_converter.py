"""PDF conversion utility for figures"""
import os
from PIL import Image


def convert_all_to_pdf():
    """Convert all PNG figures to PDF format"""
    print("=" * 80)
    print("CONVERTING ALL FIGURES TO PDF FORMAT")
    print("=" * 80)

    png_files = [f for f in os.listdir('figures') if f.endswith('.png')]
    png_files.sort()

    print(f"\nFound {len(png_files)} PNG files to convert\n")

    for png_file in png_files:
        png_path = f"figures/{png_file}"
        pdf_path = f"figures/{png_file.replace('.png', '.pdf')}"

        img = Image.open(png_path)

        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img

        img.save(pdf_path, 'PDF', resolution=300.0, quality=100)

        print(f"Converted: {png_file} → {pdf_path.split('/')[-1]}")

    print("\n" + "=" * 80)
    print(f"ALL {len(png_files)} FIGURES CONVERTED TO PDF")
    print("=" * 80)
    print("\nBoth PNG and PDF versions available in ./figures/")


if __name__ == "__main__":
    convert_all_to_pdf()
