import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
       
        self.linear_layers = nn.Sequential(
            #apply flatten layer?
            #https://programmerall.com/article/8686604124/
        nn.Flatten(),
            #Dropout should not be included when performing on validation data
            #Tune Dropout parameter for performance, lower it and check performance
            #https://nnart.org/should-you-use-dropout/
            #https://discuss.pytorch.org/t/what-is-the-difference-between-nn-dropout2d-and-nn-dropout/108192
            

#         nn.Linear(in_features=512*7*7, out_features=4096),
#         nn.LeakyReLU(),
#             #nn.Dropout(0.3),
#         nn.Linear(in_features=4096, out_features=4096),
#         nn.LeakyReLU(),
#             #nn.Dropout(0.3),
#         nn.Linear(in_features=4096, out_features=1),

        nn.Linear(in_features=512*7*7, out_features=500),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=500, out_features=500),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=500, out_features=1)
# 1960aca70cc080d2575d221d6bf6f58b6cdad5df
        )

    def forward(self, x):
        out = self.features(x) #conv layers
        #out = out.view(out.size(0), -1) #flatten layer
        out = self.linear_layers(out) #fully-connected layers
        #out = out.view(out.size(0), -1) #flatten layer
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.LeakyReLU()]
                in_channels = x
        return nn.Sequential(*layers)

