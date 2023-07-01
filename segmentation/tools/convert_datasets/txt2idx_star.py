import pickle
import os


def load_register(txt2idx_star_pth: str):
    '''Returns the specified txt_star2_idx_str register dict object from disk,
    or create an empty dict object if the file does not exist.
    '''
    if not os.path.isfile(txt2idx_star_pth):
        txt2_idx_str = {}
        return txt2_idx_str

    with open(txt2idx_star_pth, 'rb') as f:
        txt2_idx_str = pickle.load(f)

    return txt2_idx_str


def save_register(txt2idx_star_path: str, txt2idx_star):
    try:
        with open(txt2idx_star_path, 'wb') as f:
            f.write(pickle.dumps(txt2idx_star))
    except IOError:
        print(f'Error saving \'txt2idx_star\' dict to {txt2idx_star_path}')


def add_entry(txt2idx_star: dict, txt: str):
    '''Add a new text entry with a unique index for new entries.
    Args:
        txt_star2idx_star: Registrered text strings with unique indices.
        txt: Text string.
    Returns:
        Updated dictionary.
    '''
    # Text string already exists
    if txt in txt2idx_star.keys():
        return txt2idx_star
    # Add text string
    idx = len(txt2idx_star)
    txt2idx_star[txt] = idx
    return txt2idx_star
