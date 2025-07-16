import torch
import torch.onnx
import torch.nn as nn

import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Verwende: {device}")

class ONNXExporter:
    def __init__(self, model, input_dim, export_path="generator.onnx", device="cuda"):
        self.model = model.to(device).eval()
        self.input_dim = input_dim
        self.export_path = export_path
        self.device = device

    def export(self):
        dummy_input = torch.randn(1, self.input_dim, device=self.device)

        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

        torch.onnx.export(
            self.model,
            dummy_input,
            self.export_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            export_params=True
        )

        print(f"ONNX-Modell erfolgreich exportiert nach: {self.export_path}")


class Generator(nn.Module):
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
            self._make_layer(128, 64),  # 16->32
            self._make_layer(64, 32),  # 32->64
            self._make_layer(32, 16),  # 64->128
            self._make_layer(16, 8),  # 128->256
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

def load_checkpoint(path):
    """Gespeichertes Model laden"""
    if not os.path.exists(path):
        print(f"Checkpoint {path} nicht gefunden!")
        return None

    checkpoint = torch.load(path, map_location=device)

    G = Generator().to(device)
    G.load_state_dict(checkpoint['generator'])
    G.eval()

    print(f"Model geladen von Epoch {checkpoint['epoch']}")
    return G

# Beispiel: Modell laden
G = load_checkpoint("training/StyleGANv2/epochen/checkpoint_0430.pth")

# Exporter verwenden
exporter = ONNXExporter(model=G, input_dim=512, export_path="ganGenerator.onnx", device="cuda")
exporter.export()