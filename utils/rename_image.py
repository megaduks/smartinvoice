from PIL import Image
import imagehash
import os, sys


def main(file_name: str) -> str:
    """Renames file provided by the filename using hash computed from file's content

    :param file_name: input file name
    :return: new file name generated from file content hash
    """

    if not os.path.exists(file_name):
        raise FileNotFoundError

    fname, extension = file_name.split('.')
    hash_file_name = f'{imagehash.average_hash(Image.open(file_name))}.{extension}'
    os.rename(file_name, hash_file_name)

    return hash_file_name


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Usage: python rename_image.py file1 file2 ...")
        sys.exit(0)
    else:
        for file in sys.argv[1:]:
            main(file)