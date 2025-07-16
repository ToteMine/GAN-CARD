import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

class ONNXImageGenerator:
    def __init__(self, model_path, input_dim=512, device='cuda'):
        self.model_path = model_path
        self.input_dim = input_dim

        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name

    def generate(self, batch_size=1):
        input_data = np.random.randn(batch_size, self.input_dim).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: input_data})
        return outputs[0]

    def generate_and_show(self, batch_size=1):
        images = self.generate(batch_size)

        for i in range(batch_size):
            img = images[i]

            # Annahme: img hat Form (C, H, W), wir brauchen (H, W, C)
            img = np.transpose(img, (1, 2, 0))

            # Normalisierung:
            # Manche GANs geben Werte in [-1, 1], wir skalieren auf [0,1]
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)

            plt.figure(figsize=(4,4))
            plt.axis('off')
            plt.title(f'Generated Image {i+1}')
            plt.imshow(img)
            plt.show()


# Beispiel:
generator = ONNXImageGenerator("../../../Workspace Desktop/GAN/ganGenerator.onnx", input_dim=512, device='cuda')
generator.generate_and_show(batch_size=2)
