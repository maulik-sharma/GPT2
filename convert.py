import os
import warnings
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Suppress EbookLib's warning spam about malformed EPUBs
warnings.filterwarnings('ignore', category=UserWarning, module='ebooklib')


def epub_to_text(epub_path):
    """Reads an EPUB and extracts plain text from all document items."""
    book = epub.read_epub(epub_path)
    text_content = []

    # Iterate through all items in the EPUB
    for item in book.get_items():
        # Look only for document items (HTML/XHTML files inside the EPUB)
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse the HTML content and extract text
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')

            # Use newline as separator to preserve basic paragraph structure
            text = soup.get_text(separator='\n', strip=True)
            if text:
                text_content.append(text)

    # Join all chapters with distinct spacing
    return '\n\n---\n\n'.join(text_content)


def batch_convert(input_dir, output_dir):
    """Converts all EPUBs in the input directory to TXT in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Process every file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.epub'):
            epub_path = os.path.join(input_dir, filename)

            # Create the new .txt filename
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)

            print(f"Converting: {filename}...")
            try:
                text = epub_to_text(epub_path)
                # Save the extracted text using UTF-8 encoding
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"  -> Saved {txt_filename}")
            except Exception as e:
                print(f"  -> Failed to convert {filename}: {e}")


if __name__ == '__main__':
    # Set your directories here
    INPUT_DIR = './data/epub'
    OUTPUT_DIR = './data/txt'

    batch_convert(INPUT_DIR, OUTPUT_DIR)