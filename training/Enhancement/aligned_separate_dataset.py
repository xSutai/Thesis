import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

# Portions of this code were generated with the assistance of ChatGPT (OpenAI, 2025) and subsequently modified by the author.
# [1]OpenAI. (2025). ChatGPT (May 2025 version) [Large language model]. https://chat.openai.com 

class AlignedSeparateDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'A')  # e.g. test/A
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'B')  # e.g. test/B

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        self.transform_A = get_transform(opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(opt, grayscale=(self.output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return min(len(self.A_paths), len(self.B_paths))

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

