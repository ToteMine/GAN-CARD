import torch
import torch.nn as nn
import os
from pathlib import Path
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_PATH = Path(__file__).parent


class Generator_OLD(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Start: 512 -> 4x4
        self.start = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # Upsampling Layers: 4x4 -> 256x256
        self.ups = nn.ModuleList([
            self._make_layer(512, 256),  # 4->8
            self._make_layer(256, 128),  # 8->16
            self._make_layer(128, 64),   # 16->32
            self._make_layer(64, 32),    # 32->64
            self._make_layer(32, 16),    # 64->128
            self._make_layer(16, 8),     # 128->256
        ])
        
        # Final RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(8, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
        # Gewichte initialisieren
        self.apply(self._init_weights)
    
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        x = z.view(z.size(0), 512, 1, 1)
        x = self.start(x)
        
        for up in self.ups:
            x = up(x)
        
        return self.to_rgb(x)
    

def generate_image_from_checkpoint(ckpt_path,
                                    n_samples=1,
                                    z=None):
    G = Generator_OLD().to(device)

    # Latent-Vektor vorbereiten
    if z is None:
        z = torch.randn(n_samples, 512, device=device)
    else:
        assert isinstance(z, torch.Tensor), "z muss ein Tensor sein"
        assert z.shape == (n_samples, 512), f"z muss die Form [{n_samples}, 512] haben"

    # Checkpoint laden
    checkpoint = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(checkpoint['generator'])
    G.eval()

    with torch.no_grad():
        fake_images = G(z)

    # Erstes Bild nehmen, normalisieren von [-1,1] auf [0,1]
    img_tensor = (fake_images[0].clamp(-1, 1) + 1) / 2

    # In PIL umwandeln
    to_pil = transforms.ToPILImage()
    return to_pil(img_tensor.cpu())


def main(seed = None)->list:
    if seed is None:
        seed = torch.randn(1, 512, device=device)
    img = generate_image_from_checkpoint(os.path.join(OUTPUT_PATH, "checkpoint_0430.pth"), z=seed)
    return seed, img


if __name__ == '__main__':
    main()
    