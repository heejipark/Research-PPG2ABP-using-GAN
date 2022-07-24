"""
## Model
1. Generator(Encoder)
    -> C64, C128, C256, C512, C512, C512, C512, C512
2. Generator(Decoder)
    -> CD512 - CD 
3. Discriminator
    -> C64 - C128 - C256 - C512 - C1
"""
import torch.nn as nn
import torch.nn.functional as F

# ResidualBlock --------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        layer = [nn.ReflectionPad1d(1),
                 nn.Conv1d(in_features, in_features, 3),
                 nn.InstanceNorm1d(in_features),
                 nn.ReLU(inplace=True),
                 nn.ReflectionPad1d(1),
                 nn.Conv1d(in_features, in_features, 3),
                 nn.InstanceNorm1d(in_features)]
        
        self.conv_block = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.conv_block(x)

# Generator --------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
    
        # Initial convolution block       
        self.model1 = nn.Sequential(
                 nn.ReflectionPad1d(3),
                 nn.Conv1d(input_nc, 64, 7),
                 nn.InstanceNorm1d(64),
                 nn.ReLU(inplace=True)
                 )

        # Encoder - Downsampling
        in_features = 64
        out_features = in_features*2
        model2 = []
        for _ in range(2):
            model2 += [nn.Conv1d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                       nn.InstanceNorm1d(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2
        self.model2 = nn.Sequential(*model2)
        
        model3=[]
        # Residual blocks
        for _ in range(n_residual_blocks):
            model3 += [ResidualBlock(in_features)]
        self.model3 = nn.Sequential(*model3)
        
        # Decoder - Upsampling
        model4 = []
        out_features = in_features//2
        for _ in range(2):
            model4 += [nn.ConvTranspose1d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm1d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2
        self.model4 = nn.Sequential(*model4)
        
        # Output layer
        self.model5 = nn.Sequential(
                  nn.ReflectionPad1d(3),
                  nn.Conv1d(64, output_nc, 7),
                  nn.Tanh())

    def forward(self, x):
        x = self.model1(x) 
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        out = self.model5(x)
        return out

# Discriminator --------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv1d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv1d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm1d(128), 
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv1d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm1d(256), 
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv1d(256, 512, 4, padding=1),
                  nn.InstanceNorm1d(512), 
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv1d(512, 1, 4, padding=1),
                  nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool1d(x, x.size()[2:]).view(x.size()[0], -1)


