import torch
import torchvision
from torch import nn

from utils.models import autoencoder, transformer

def get_model(num_classes,
              model_name,
              num_frames):
    ''' returns the classifier '''

    if model_name == "autoencoder":
        print("Loading the autoencoder model...")
        model = autoencoder.Autoencoder(skipDropout=0.8)
    elif model_name == "autoencoder_noSkip":
        print("Loading the autoencoder model without skip connection...")
        model = autoencoder.Autoencoder(skipDropout=1.0)
    elif model_name == "endToEnd_Alexnet":
        print("Loading the autoencoder model...")
        model_backbone = torchvision.models.alexnet(pretrained=False)
        model_backbone = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0),
            model_backbone.features,
            nn.AdaptiveAvgPool2d([2,2])
            )
        model_transformer = transformer.Transformer(input_dim=256*2*2, in_channels=6, seq_len=num_frames, embed_dim=64, nhead=4, num_layers=6, num_classes=num_classes)
        model = transformer.EndToEnd(model_backbone, model_transformer)
    elif model_name == "endToEnd_Alexnet_pretrain_freeze":
        print("Loading the autoencoder model...")
        model_backbone = torchvision.models.alexnet(pretrained=True)
        model_backbone = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0),
            model_backbone.features,
            nn.AdaptiveAvgPool2d([2,2])
            )
        for param in model_backbone[1].parameters():
            param.requires_grad = False
        model_transformer = transformer.Transformer(input_dim=256*2*2, in_channels=6, seq_len=num_frames, embed_dim=64, nhead=4, num_layers=6, num_classes=num_classes)
        model = transformer.EndToEnd(model_backbone, model_transformer)
    elif model_name == "endToEnd_Resnet":
        print("Loading the autoencoder model...")
        model_backbone = torchvision.models.resnet18(pretrained=False)
        model_backbone = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0),
            model_backbone.conv1,
            model_backbone.bn1,
            model_backbone.relu,
            model_backbone.maxpool,
            model_backbone.layer1,
            model_backbone.layer2,
            model_backbone.layer3,
            model_backbone.layer4,
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d([2,2])
            )    
        model_transformer = transformer.Transformer(input_dim=256*2*2, in_channels=6, seq_len=num_frames, embed_dim=64, nhead=4, num_layers=6, num_classes=num_classes)
        model = transformer.EndToEnd(model_backbone, model_transformer)
    elif model_name == "endToEnd_Resnet_pretrain_freeze":
        print("Loading the autoencoder model...")
        model_backbone = torchvision.models.resnet18(pretrained=True)
        model_backbone = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0),
            model_backbone.conv1,
            model_backbone.bn1,
            model_backbone.relu,
            model_backbone.maxpool,
            model_backbone.layer1,
            model_backbone.layer2,
            model_backbone.layer3,
            model_backbone.layer4,
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d([2,2])
            )
        for param in model_backbone[1].parameters():
            param.requires_grad = False
        for param in model_backbone[2].parameters():
            param.requires_grad = False
        for param in model_backbone[3].parameters():
            param.requires_grad = False
        for param in model_backbone[4].parameters():
            param.requires_grad = False
        for param in model_backbone[5].parameters():
            param.requires_grad = False
        for param in model_backbone[6].parameters():
            param.requires_grad = False
        for param in model_backbone[7].parameters():
            param.requires_grad = False
        for param in model_backbone[8].parameters():
            param.requires_grad = False  
        model_transformer = transformer.Transformer(input_dim=256*2*2, in_channels=6, seq_len=num_frames, embed_dim=64, nhead=4, num_layers=6, num_classes=num_classes)
        model = transformer.EndToEnd(model_backbone, model_transformer)
    elif model_name == "endToEnd" or model_name == "endToEnd_fast" or model_name == "endToEnd_finetuning" or\
            model_name == "endToEnd_noSkip" or model_name == "endToEnd_noSkip_fast" or model_name == "endToEnd_noSkip_finetuning":
        if "_fast" in model_name:
            print("Excluding the backbone model to use the pretrained embeddings...")
            model_backbone = autoencoder.Autoencoder
        else:
            # Load the pretrained backbone model
            if "_noSkip" in model_name:
                print("Loading the end-to-end model without skip connection...")
                model_backbone = autoencoder.Autoencoder(onlyEncoder=True)
                model_backbone.load_state_dict(torch.load("./logs/path/to/chechpoint.pth"), strict=False)
            else:
                print("Loading the end-to-end model...")
                model_backbone = autoencoder.Autoencoder(onlyEncoder=True, skipDropout=0.8)
                model_backbone.load_state_dict(torch.load("./logs/path/to/chechpoint.pth"), strict=False)
        
            # Freeze the backbone model if the model is endToEnd
            if "_finetuning" not in model_name:
                print("Freezing the backbone model...")
                for param in model_backbone.parameters():
                    param.requires_grad = False
        
        # Create the transformer model
        model_transformer = transformer.Transformer(input_dim=256*2*2, in_channels=6, seq_len=num_frames, embed_dim=64, nhead=4, num_layers=6, num_classes=num_classes)
        
        # Create the end-to-end model
        model = transformer.EndToEnd(model_backbone, model_transformer)
    else:
        raise RuntimeError('Model name not found!')

    return model