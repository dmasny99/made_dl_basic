import torch
from PIL import Image


class CaptchaDataset:
    def __init__(self, img_list, char_dict, inv_char_dict, transform=None):
        self.img_list = img_list
        self.transform = transform
        self.char_dict = char_dict
        self.inv_char_dict = inv_char_dict
        
    def __len__(self):
        return len(self.img_list)

    def __map_labelt_to_nums(self, label):
        num_label = []
        for ch in label:
            num_label.append(self.char_dict[ch])
        return num_label
            
    def __getitem__(self, idx):
        data = self.img_list[idx]
        image = Image.open(data).convert('L')
        file_name = data.split("/")[-1].removesuffix(".png")
        label = self.__map_labelt_to_nums(file_name)
        label = torch.tensor(label)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label