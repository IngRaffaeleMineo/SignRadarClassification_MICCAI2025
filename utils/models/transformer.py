import torch
import torch.nn as nn
import math

from utils.models import autoencoder

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, in_channels, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=(1, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, embed_dim)
        )

    def forward(self, x):
        batch, inputs, time, ch, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        x = x.view(batch * time, inputs, ch, h, w)
        x = self.conv(x).contiguous()
        x = x.view(batch, time, 1, ch, h, w)
        x = x.view(batch, time, 1 * ch * h * w)
        x = self.proj(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim=256):
        super().__init__()
        pos_embedding_vector = torch.empty(1, seq_len+1, embed_dim)
        
        val = math.sqrt(6. / float(embed_dim))
        nn.init.uniform_(pos_embedding_vector, -val, val) # xavier_uniform initialization
        
        self.pos_embedding = nn.Parameter(pos_embedding_vector)

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :]
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, in_channels, seq_len, embed_dim, nhead, num_layers, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_dim=input_dim, in_channels=in_channels, embed_dim=embed_dim)
        self.pos_embed = PositionalEmbedding(seq_len=seq_len, embed_dim=embed_dim)
        
        cls_token_vector = torch.empty(1, 1, embed_dim)
        val = math.sqrt(6. / float(embed_dim))
        nn.init.uniform_(cls_token_vector, -val, val) # xavier_uniform initialization
        self.cls_token = nn.Parameter(cls_token_vector)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_embed(x)
        enc = self.transformer_encoder(x)
        
        cls_output = enc[:, 0]
        logits = self.classifier(cls_output)
        
        return logits


class EndToEnd(nn.Module):
    def __init__(self, backbone, transformer : Transformer):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.input_preprocessed = False
        self.len_x = None
        self.len_batch = None
        self.len_ch_time = None
        self.len_time = None
    
    def preprocess_input(self, x):
        x, (self.len_x, self.len_batch, self.len_ch_time, self.len_time) = autoencoder.Autoencoder.preprocess_input(x)
        self.input_preprocessed = True
        return x
    
    def forward(self, x):
        if isinstance(self.backbone, type): # if backbone is a class, you are in fast mode (use pretrainded embeddings)
            out2 = x
        else:
            if not self.input_preprocessed:
                raise ValueError("Input not preprocessed. Use preprocess_input() method before forward pass.")
            out = self.backbone(x) # batch * radar_inputs*time, feature=256, height=2, width=2
            out2 = autoencoder.Autoencoder.postprocess_embedding(out, *(self.len_x, self.len_batch, self.len_ch_time, self.len_time))
            self.input_preprocessed = False # reset the flag
        
        out3 = self.transformer(out2)
        
        return out3
    

if __name__ == "__main__":
    model = Transformer(input_dim=256*2*2, in_channels=6, seq_len=48, embed_dim=64, nhead=4, num_layers=6, num_classes=126)
    model = EndToEnd(backbone=autoencoder.Autoencoder, transformer=model)
    x = torch.randn(8, 6, 48, 256, 2, 2) # batch, radar_inputs, time, feature, height, width
    output = model(x)
    print(x.shape, "->", output.shape)
    
    model = EndToEnd(backbone=autoencoder.Autoencoder(onlyEncoder=True), transformer=model)
    x = [torch.randn(2, 48, 128, 128) for _ in range(6)]
    try:
        _ = model(x)
    except ValueError as e:
        print(e, "OK")
    finally:
        x2 = model.preprocess_input(x)
    out = model(x2)
    print(len(x), "x", x[0].shape, "->", out.shape)