import numpy as np
from pathlib import Path
from torchvision import transforms
import onnxruntime as ort
import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).parent


class ONNXImageGenerator:
    def __init__(self, model_path, input_dim=512, device='cuda'):
        self.model_path = model_path
        self.input_dim = input_dim

        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name

    def generate(self, seed):
        # seed muss Shape (batch_size, input_dim) haben, z.B. (1, 512)
        output = self.session.run(None, {self.input_name: seed})
        img = output[0][0]  # Erstes Bild aus Batch, Shape (C, H, W)

        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        img = (img + 1) / 2  # von [-1,1] auf [0,1]
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)  # PIL erwartet uint8

        to_pil = transforms.ToPILImage()
        return to_pil(img)


def main(seed=None):
    # Batch Size = 1, Seed zufällig falls None
    if seed is None:
        seed = np.random.randn(1, 512).astype(np.float32)

    generator = ONNXImageGenerator("ganGenerator.onnx", input_dim=512, device='cuda')
    img = generator.generate(seed)
    return seed, img


if __name__ == '__main__':
    main()
