import torch
import torch.nn as nn
from .base_model import BaseModel
from .real_full_restormer_Ch import Restormer
from pytorch_msssim import ssim

# Portions of this code were generated with the assistance of ChatGPT (OpenAI, 2025) and subsequently modified by the author.
# [1]OpenAI. (2025). ChatGPT (May 2025 version) [Large language model]. https://chat.openai.com 


class RestormerModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='aligned_separate', norm='none', netG='restormer')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=10.0)
            #parser.add_argument('--lambda_SSIM', type=float, default=100.0)
            #pass
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['L1', 'SSIM']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G']

        self.netG = RestormerWrapper(opt.input_nc, opt.output_nc).to(self.device)
        #use multiple gpus
        if torch.cuda.device_count() > 1:
           print(f" Using {torch.cuda.device_count()} GPUs with DataParallel")
           self.netG = nn.DataParallel(self.netG)

        

        if self.isTrain:
            self.criterionL1 = nn.L1Loss()
            self.criterionSSIM = ssim
            self.lambda_L1 = opt.lambda_L1
            self.lambda_SSIM = opt.lambda_SSIM
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_G(self):
        self.loss_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        self.loss_SSIM = (1 - self.criterionSSIM(self.fake_B, self.real_B, data_range=1.0, size_average=True)) * self.lambda_SSIM
        self.loss_G = self.loss_L1 + self.loss_SSIM
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

class RestormerWrapper(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.model = Restormer(inp_channels=input_nc, out_channels=output_nc)
        self.output_layer = nn.Conv2d(3, output_nc, kernel_size=1)

    def forward(self, x):
        out = self.model(x)
        return self.output_layer(out)

