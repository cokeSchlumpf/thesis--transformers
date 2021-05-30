import hashlib
import pickle

def checksum_of_file(file: str) -> str:
    """
    Calculates the MD5 checksum of a file. The file must exist.

    Args:
        file: The path of the file.

    Returns:
        The checksum of the file.
    """
    md5 = hashlib.md5()

    f = open(file, "rb")
    while chunk := f.read(4096):
        md5.update(chunk)

    return md5.hexdigest()


def read_file_to_object(file: str) -> any:
    """
    Reads a file and interprets its content as a Python object (with pickle). The file must exist.
    Args:
        file: The file to read.

    Returns:
        The Python object.
    """

    with open(file, 'rb') as f:
        return pickle.load(f)


def read_file_to_string(file: str) -> str:
    """
    Reads a text file and returns its contents. If file does not exist, an empty string is returned.

    Args:
        file: The file to read.

    Returns:
        The content of the file.
    """

    try:
        with open(file, 'r') as f:
            data = f.read().replace('\n', '')

        return data
    except IOError:
        return ''


def write_object_to_file(file: str, obj: any) -> None:
    """
    Writes the Python object to a file. If the file exists, its content will be overridden.

    Args:
        file: The file to wrote to
        obj: The object to save

    Returns:
        Nothing.
    """

    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def write_string_to_file(file: str, content: str) -> None:
    """
    Writes the content of the string to the file. If file exists its content will be overridden.

    Args:
        file: The file to write to.

    Returns:
        Nothing.
    """

    with open(file, 'w') as f:
        f.write(content)
