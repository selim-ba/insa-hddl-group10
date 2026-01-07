import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights

# Binary classification models

class CNN_Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512, 512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) #global average pooling
        self.fc2 = nn.Linear(512, 2)  # binary classif

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112 -> 56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56 -> 28
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 28 --> 14

        x = self.gap(x) # [B, 512, 1, 1]                       
        x = x.view(x.size(0), -1) # [B, 512]            
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class ResNet18_Binary(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = resnet18(weights=weights)
        else :
            self.backbone = resnet18(weights=None)

        # we replace ResNet's final fully connected layer by a binary classif. head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256,2)
        )

    def forward(self,x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """
            to freeze all backbone layers except the classification head
        """
        for name, module in self.backbone.named_modules():
            if not name.startswith("fc"):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()  # set BatchNorm layers to eval mode
                    
        for name, p in self.backbone.named_parameters():
            if not name.startswith("fc."): #to keep head trainable
                p.requires_grad = False

    def unfreeze_backbone(self):
        """
            to unfreeze the backbone
        """
        for p in self.backbone.parameters():
            p.requires_grad = True
    
    def trainable_parameters(self):
        """
            returns the parameters that require gradients
        """
        return filter(lambda p: p.requires_grad, self.parameters())

# Multiclass classification models

class CNN_Multiclass(nn.Module):
    def __init__(self, num_classes=37):
        super(CNN_Multiclass, self).__init__()
        
        # Conv blocks matching Keras architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input: 224x224x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output: 224x224x64
        self.pool1 = nn.MaxPool2d(2, 2)                          # Output: 112x112x64
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Output: 112x112x128
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Output: 112x112x256
        self.pool2 = nn.MaxPool2d(2, 2)                          # Output: 56x56x256
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate flatten size: 56x56x256 = 802816
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(56 * 56 * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv block 1 (matching Keras Conv2D(32) + Conv2D(64) + MaxPool)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv block 2 (enhanced for color images)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # FC layers (matching Keras Dense(128) + Dense(num_classes))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Raw logits (softmax in loss)
        
        return x
    
class VGG16_Multiclass(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        # backbone.features are convolutional layers
        # backbone.classifier is the fully-connected head for ImageNet

        # Replace the last classifier layer to match num_classes = 37 breeds
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        # freeze all except the last classifier layer
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.classifier[-1].parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        # unfreeze everything for fine-tuning
        for p in self.backbone.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        # only the head params when backbone is frozen
        return [p for p in self.parameters() if p.requires_grad]

# Segmentation models
# U-Net
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class DownSampleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = double_conv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        x_skip = x
        x = self.maxpool(x)
        return x_skip, x

class UpSampleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = double_conv(in_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.upconv(x)
        x = torch.cat((x_skip, x), dim=1)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.downsample1 = DownSampleUNet(in_channels, 64)
        self.downsample2 = DownSampleUNet(64, 128)
        self.downsample3 = DownSampleUNet(128, 256)
        self.downsample4 = DownSampleUNet(256, 512)

        self.bottleneck = double_conv(512, 1024)

        self.upsample1 = UpSampleUNet(1024, 512)
        self.upsample2 = UpSampleUNet(512, 256)
        self.upsample3 = UpSampleUNet(256, 128)
        self.upsample4 = UpSampleUNet(128, 64)

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1_skip, x1 = self.downsample1(x)
        x2_skip, x2 = self.downsample2(x1)
        x3_skip, x3 = self.downsample3(x2)
        x4_skip, x4 = self.downsample4(x3)

        x_bottleneck = self.bottleneck(x4)

        x = self.upsample1(x_bottleneck, x4_skip)
        x = self.upsample2(x, x3_skip)
        x = self.upsample3(x, x2_skip)
        x = self.upsample4(x, x1_skip)

        output = self.last_conv(x)
        return output

# U-Net++
class ConvBlock(nn.Module):
    """"The double convolution block is the same as U-Net's one"""
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return self.convblock(x)
    
class UNetPP(nn.Module):
    """
        xi_j = H(xi-1_j) si j = 0
        xi_j = H([ x^{i,0}, ..., x^{i,j-1}, up(x^{i+1,j-1}) ]) si j > 0

        Note : without deep supervision
    """
    def __init__(self,in_channels=3,num_classes=3,filters=(32,64,128,256,512)):
        super().__init__()

        f0, f1, f2, f3, f4 = filters
        
        # Encoder : x0_0, x1_0, x2_0, x3_0, x4_0
        self.conv0_0 = ConvBlock(in_channels,f0) # x0_0 = H_0^0(input)
        self.conv1_0 = ConvBlock(f0,f1) # x1_0 = H_1^0(pool(x0_0))
        self.conv2_0 = ConvBlock(f1,f2) # x2_0 = H_2^0(pool(x1_0))
        self.conv3_0 = ConvBlock(f2,f3) # x3_0 = H_3^0(pool(x2_0))
        self.conv4_0 = ConvBlock(f3,f4) # x4_0 = H_4^0(pool(x3_0))
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Upsampling
        self.up1 = nn.ConvTranspose2d(f1,f0,2,stride=2)
        self.up2 = nn.ConvTranspose2d(f2,f1,2,stride=2)
        self.up3 = nn.ConvTranspose2d(f3,f2,2,stride=2)
        self.up4 = nn.ConvTranspose2d(f4,f3,2,stride=2)


        # Decoder : for each H_i^j : concatenation of x_i^0, ..., x_i^{j-1} and up(x_{i+1}^{j-1})
        # j = 1 : x0_1, x1_1, x2_1, x3_1
        self.conv0_1 = ConvBlock(f0 + f0, f0)   # [x0_0, up(x1_0)] -> 2*f0
        self.conv1_1 = ConvBlock(f1 + f1, f1)   # [x1_0, up(x2_0)] -> 2*f1
        self.conv2_1 = ConvBlock(f2 + f2, f2)   # [x2_0, up(x3_0)] -> 2*f2
        self.conv3_1 = ConvBlock(f3 + f3, f3)   # [x3_0, up(x4_0)] -> 2*f3

        # j = 2 : x0_2, x1_2, x2_2
        self.conv0_2 = ConvBlock(f0*3, f0)      # [x0_0, x0_1, up(x1_1)] -> 3*f0
        self.conv1_2 = ConvBlock(f1*3, f1)      # [x1_0, x1_1, up(x2_1)] -> 3*f1
        self.conv2_2 = ConvBlock(f2*3, f2)      # [x2_0, x2_1, up(x3_1)] -> 3*f2

        # j = 3 : x0_3, x1_3
        self.conv0_3 = ConvBlock(f0*4, f0)      # [x0_0, x0_1, x0_2, up(x1_2)] -> 4*f0
        self.conv1_3 = ConvBlock(f1*4, f1)      # [x1_0, x1_1, x1_2, up(x2_2)] -> 4*f1

        # j = 4 : x0_4
        self.conv0_4 = ConvBlock(f0*5, f0)      # [x0_0, x0_1, x0_2, x0_3, up(x1_3)] -> 5*f0

        # final classifier on x0_4 (no deep supervision)
        self.final = nn.Conv2d(f0,num_classes,kernel_size=1)

    def forward(self,x):

        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Skip + Decoder #  xi_j = H([ x^{i,0}, ..., x^{i,j-1}, up(x^{i+1,j-1}) ]) si j > 0
        # j = 1
        x0_1 = self.conv0_1( torch.cat([x0_0,self.up1(x1_0)], dim=1 ))
        x1_1 = self.conv1_1( torch.cat([x1_0,self.up2(x2_0)], dim=1 ))
        x2_1 = self.conv2_1( torch.cat([x2_0,self.up3(x3_0)], dim=1 ))
        x3_1 = self.conv3_1( torch.cat([x3_0,self.up4(x4_0)], dim=1 ))

        # j = 2
        x0_2 = self.conv0_2( torch.cat([x0_0, x0_1, self.up1(x1_1)], dim=1 ))
        x1_2 = self.conv1_2( torch.cat([x1_0, x1_1, self.up2(x2_1)], dim=1 ))
        x2_2 = self.conv2_2( torch.cat([x2_0, x2_1, self.up3(x3_1)], dim=1 ))

        # j = 3
        x0_3 = self.conv0_3( torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2)], dim=1 ))
        x1_3 = self.conv1_3( torch.cat([x1_0, x1_1, x1_2, self.up2(x2_2)], dim=1 ))

        # j = 4
        x0_4 = self.conv0_4( torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1(x1_3)], dim=1 ) )

        output = self.final(x0_4)
        return output

