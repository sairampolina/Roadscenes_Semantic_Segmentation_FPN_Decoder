
import torch
import torch.nn as nn

import torchvision.models as models

import torch.nn.functional as F

from typing import Iterable


### Encoder class ###

class Encoder(nn.Module):

    def __init__(self, encoder_type: str = "resnet18"):
        super().__init__()
        
        assert hasattr(models, encoder_type), "Invalid Encoder type"

        self.feature_extractor = getattr(models, encoder_type)(weights = "IMAGENET1K_V1")

 
    def forward(self, input):

        x = self.feature_extractor.conv1(input)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)

        layer1_out = self.feature_extractor.layer1(x)
        layer2_out = self.feature_extractor.layer2(layer1_out)  
        layer3_out = self.feature_extractor.layer3(layer2_out)       
        layer4_out = self.feature_extractor.layer4(layer3_out)

        return  layer4_out, layer3_out, layer2_out, layer1_out
    
    def get_num_channels(self):

        """
        return no of channels in each layer
                    layer4 -> layer1
        """

        channels = []

        layers = [getattr(self.feature_extractor,'layer'+ f"{i}") for i in range(1,5)]

        for layer in layers:
            channels.append(layer[-1].bn2.weight.size(0))

        return channels[::-1]
    

## Lateral Connection ##

class LateralConnection(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.proj = nn.Conv2d(channels_in,channels_out,kernel_size=1)
    

    def forward(self, prev, cur):
         
         up = F.interpolate(prev, size = cur.size()[-2:], mode="nearest")

         proj = self.proj(cur)

         return proj + up


## *****Decoder Block ****##

class Decoder(nn.Module):

    def __init__(self, channels_in : Iterable[int], channels_out : int = 128):

        super().__init__()
    
        self.module = nn.ModuleList()

        self.module.append(nn.Conv2d(channels_in[0], channels_out, kernel_size=1))
        
        for i in range(1, len(channels_in)):
            self.module.append(LateralConnection(channels_in[i],channels_out))

    def forward(self, x : Iterable):
        output = [self.module[0](x[0])]

        for i in range(1,len(x)):
            output.append(self.module[i](output[-1],x[i]))
        
        return output[-1]
       

    


#### Semantic Segmentation ####

class SemSeg(nn.Module):

    def __init__(self, num_classes, pyramid_channels = 128, final_upsample = False):

        super().__init__()
        self.encoder = Encoder("resnet18")
        self.decoder = Decoder(self.encoder.get_num_channels(),
                               channels_out= pyramid_channels)
        self.classifier = nn.Sequential(
            nn.Conv2d(pyramid_channels,pyramid_channels,kernel_size=3,stride =1,padding=1,bias=False),
            nn.BatchNorm2d(pyramid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(pyramid_channels,num_classes,kernel_size=1)
        )
        self.final_upsample = final_upsample

    def forward(self, x):
        encoder = self.encoder(x)
        decoder_out = self.decoder(encoder)
        classifier  = self.classifier(decoder_out)
        
        if self.final_upsample:
            classifier = F.interpolate(classifier, x.size()[-2:],mode="bilinear",align_corners=False)
        
        return classifier

