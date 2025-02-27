import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_conv = self.conv(x)
        out_bn = self.bn(out_conv)
        out_relu = self.relu(out_bn)
        return out_relu

class EncoderResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.bn_residual = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out_residual = self.residual(x)
        res = self.bn_residual(out_residual)
        
        out1 = self.conv1(x)
        out2_skip = self.conv2(out1)
        out3 = self.conv3(out2_skip)
        
        out3 = out3 + res
        
        out_relu = self.relu(out3)
        
        return out_relu, out2_skip

class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn_residual = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x, skip = x
        
        out_residual = self.residual(x)
        res = self.bn_residual(out_residual)
        
        out1 = self.conv1(x)
        out1 = torch.cat([out1, skip], dim=1)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        
        out3 = out3 + res
        
        out_relu = self.relu(out3)
        
        return out_relu

class Autoencoder(nn.Module):
    def __init__(self, onlyEncoder=False, skipDropout=0.0):
        super().__init__()
        self.onlyEncoder = onlyEncoder
        self.skipDropout = skipDropout
        
        # Encoder con downsampling
        self.enc1 = EncoderResidualBlock(1, 64)
        self.enc2 = EncoderResidualBlock(64, 128)
        self.enc3 = EncoderResidualBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=8, stride=8)
        )
        
        # Se backbone non serve il decoder
        if self.onlyEncoder:
            return
        
        # Upsample bottleneck
        self.upsampleBottleneck = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=8)
        
        # Dropout skip connections
        self.dropout = nn.Dropout(p=self.skipDropout)

        # Decoder
        self.dec3 = DecoderResidualBlock(256, 128)
        self.dec2 = DecoderResidualBlock(128, 64)
        self.dec1 = DecoderResidualBlock(64, 1)

        # Upsample layers
        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4)
        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4)
        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)

    def forward(self, x):
        # Encoder
        e1, skip1 = self.enc1(x) 
        e2, skip2 = self.enc2(e1)
        e3, skip3 = self.enc3(e2)
        
        # Bottleneck
        embedding = self.bottleneck(e3)
        
        # Se backbone non serve il decoder
        if self.onlyEncoder:
            return embedding
        
        # Upsample bottleneck
        embedding_up = self.upsampleBottleneck(embedding)
        
        # Dropout skip connections
        skip3 = self.dropout(skip3)
        skip2 = self.dropout(skip2)
        skip1 = self.dropout(skip1)
        
        # Decoder
        d3 = self.upsample3(embedding_up)
        d3 = self.dec3([d3, skip3])

        d2 = self.upsample2(d3)
        d2 = self.dec2([d2, skip2])

        d1 = self.upsample1(d2)
        out = self.dec1([d1, skip1])

        return out, embedding
    
    @staticmethod
    def preprocess_input(x):
        len_x = None
        if isinstance(x, list):
            len_x = len(x)
            x2 = torch.stack(x, dim=0)
        else:
            x2 = x
        
        len_batch = None
        len_ch_time = None
        len_time = None
        if x2.dim() == 4:
            len_batch = x2.shape[0]
            len_ch_time = x2.shape[1]
            x3 = x2.view(len_batch * len_ch_time, *x2.shape[2:]).unsqueeze_(1)
        elif x2.dim() == 5:
            if len_x is None:
                len_batch = x2.shape[0]
                len_time = x2.shape[2]
                x3 = torch.moveaxis(x2, 2, 1).contiguous().view(len_batch * len_time, x2.shape[1], *x2.shape[3:])
            else:
                len_batch = x2.shape[1]
                len_time = x2.shape[2]
                x3 = x2.contiguous().view(len_x * len_batch * len_time, *x2.shape[3:]).unsqueeze_(1)
        elif x2.dim() > 5:
            raise ValueError("Input tensor must have 4 or 5 dimensions: got {} dimensions".format(x2.dim()))
        elif x2.dim() < 4:
            raise ValueError("Input tensor must have 4 or 5 dimensions: got {} dimensions".format(x2.dim()))
        else:
            raise RuntimeError("Path not reachable")
        
        return x3.clone().detach(), (len_x, len_batch, len_ch_time, len_time)
    
    @staticmethod
    def postprocess_output(out, len_x, len_batch, len_ch_time, len_time):
        out = out.clone().detach()
        
        if len_ch_time is not None: # input lista di tensori a 4 dims
            out2 = out.squeeze_(1).view(len_batch, len_ch_time, *out.shape[1:])
        elif len_time is not None: # input lista di tensori a 5 dims
            if len_x is None:
                out2 = torch.moveaxis( out.view(len_batch, len_time, *out.shape[1:]), 1, 2)
            else:
                out2 = out.squeeze_(1).contiguous().view(len_x, len_batch, len_time, *out.shape[1:])
        else:
            raise RuntimeError("Path not reachable")
            
        if len_x is None:
            out3 = out2
        else:
            out3 = [out2[i].clone().detach() for i in range(len_x)]
        
        return out3
    
    @staticmethod
    def postprocess_embedding(out, len_x, len_batch, len_ch_time, len_time):
        if len_ch_time is not None: # input lista di tensori a 4 dims
            out2 = out.squeeze_(1).view(len_batch, len_ch_time, *out.shape[1:])
        elif len_time is not None: # input lista di tensori a 5 dims
            if len_x is None:
                out2 = torch.moveaxis( out.view(len_batch, len_time, *out.shape[1:]), 1, 2)
            else:
                out2 = torch.moveaxis( out.squeeze_(1).contiguous().view(len_x, len_batch, len_time, *out.shape[1:]), 0, 1)
        else:
            raise RuntimeError("Path not reachable")
        
        return out2


if __name__ == "__main__":
    model = Autoencoder()
    
    x = [torch.randn(1, 1, 256, 1024),
         torch.randn(1, 16, 64, 256),
         torch.randn(2, 1, 256, 1024),
         torch.randn(2, 16, 64, 256)
    ]
         
    for x1 in x:
        x2, (len_x, len_batch, len_ch_time, len_time) = model.preprocess_input(x1)
        out1 = model(x2)[0]
        out2 = model.postprocess_output(out1, *(len_x, len_batch, len_ch_time, len_time))
        print(x1.shape, "->", out2.shape, "OK" if (x1.shape == out2.shape) else "KO")
    
    x = [[torch.randn(1, 1, 128, 512) for _ in range(6)],
         [torch.randn(1, 4, 64, 256) for _ in range(6)],
         [torch.randn(2, 1, 128, 512) for _ in range(6)],
         [torch.randn(2, 4, 64, 256) for _ in range(6)]]
    
    print("---")
    
    for x1 in x:
        x2, (len_x, len_batch, len_ch_time, len_time) = model.preprocess_input(x1)
        out1 = model(x2)[0]
        out2 = model.postprocess_output(out1, *(len_x, len_batch, len_ch_time, len_time))
        print(len(x1), x1[0].shape, "->", len(out2), out2[0].shape, "OK" if ( (len(x1) == len(out2)) and (x1[0].shape == out2[0].shape) ) else "KO")
    
    print("---")
    
    x1 = torch.randn(2, 48, 128, 128)
    x2, (len_x, len_batch, len_ch_time, len_time) = model.preprocess_input(x1)
    out1 = model(x2)[0]
    out2 = model.postprocess_output(out1, *(len_x, len_batch, len_ch_time, len_time))
    print(x1.shape, "->", out2.shape, "OK" if (x1.shape == out2.shape) else "KO")
    
    print("---")
    
    x1 = [torch.randn(2, 48, 128, 128) for _ in range(6)]
    x2, (len_x, len_batch, len_ch_time, len_time) = model.preprocess_input(x1)
    out1 = model(x2)[1]
    out2 = model.postprocess_embedding(out1, *(len_x, len_batch, len_ch_time, len_time))
    print("6*", x1[0].shape, "->", "Embedding shape:", out1.shape, "Embedding postprocessed shape:", out2.shape)