import os


def decode_nums(preds, inv_ch_dict):
    """
    Decode nums (predicted) to chars
    """
    batch_res = []
    for elem in preds:
        res = ""
        for ch in elem:
            if ch.item() == 0:
                continue
            else:
                res += inv_ch_dict[ch.item()]
        batch_res.append(res)
    return batch_res


def create_char_dicts(path_to_data):
    char_dict = {}
    inverse_char_dict = {}
    idx = 1
    for elem in os.listdir(path_to_data):
        elem = elem.removesuffix(".png")
        for ch in elem:
            if ch not in char_dict:
                char_dict[ch] = idx
                inverse_char_dict[idx] = ch
                idx += 1
    # define blank elem
    char_dict["-"] = 0
    inverse_char_dict[0] = "-"
    return char_dict, inverse_char_dict
        