from PIL import Image
from .base_dataset import BaseDataset, get_params, get_transform
from data.prepare_data import make_dataset
import torch


class SpectramDataset(BaseDataset):

    def __init__(self, opt):
        dataset = make_dataset(opt.data_dir, opt.max_dataset_size)
        self.audio_paths = dataset['audio']
        self.image_paths = dataset['image']
        self.labels = dataset['label']
        self.opt = opt

    def __getitem__(self, index):
        ### input A (real audio)
        audio_path = self.audio_paths[index]
        spectrogram = Image.open(audio_path).convert('RGB')
        params_spectrogram = get_params(self.opt, spectrogram.size)
        transform_image = get_transform(self.opt, params_spectrogram)
        audio_tensor = transform_image(spectrogram)

        ### input B (real images)
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        input_dict = {'audio': audio_tensor, 'image': image_tensor, 'path': audio_path}

        if not self.opt.no_label:
            label = self.labels[index]
            label_tensor = torch.FloatTensor(label)
            input_dict.__setitem__('label', label_tensor)

        return input_dict

    def __len__(self):
        return len(self.audio_paths)

    def name(self):
        return 'SpectramDataset'
