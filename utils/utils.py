import os
import matplotlib.pyplot as plt

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

def plot_predictions(model, test_iterator, inv_char_dict, device):
    test_data, test_labels = next(test_iterator)
    out, input_lengths, target_lengths = model(test_data.to(device))
    out = out.permute(1, 0, 2)
    out = out.log_softmax(2)
    out = out.argmax(-1)
    decoded_out = decode_nums(out, inv_char_dict)
    decoded_labels = decode_nums(test_labels, inv_char_dict)

    fig, ax = plt.subplots(2, 4, figsize=(16, 3))
    for i in range(2):
        for j in range(4):
            ax[i][j].imshow(test_data[2 * i + j].cpu().detach().numpy().squeeze())
            ax[i][j].set_title(f"Predicted: {decoded_out[2 * i + j]}---True: {decoded_labels[2 * i + j]}")
        