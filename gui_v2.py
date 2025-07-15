import webbrowser
import tkinter.messagebox
import cv2
import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import random
import string
import generator
import io
import os
import torch

from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Label, Frame
from tkinter import ttk, filedialog
from PIL import Image, ImageTk


device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = os.path.join(OUTPUT_PATH, 'assets/frame0/')

def relative_to_assets(path: str)->Path:
    print(ASSETS_PATH / Path(path))
    return ASSETS_PATH / Path(path)


def hide_seed_tensor_in_image(image, seed_tensor):
    """Versteckt den 512-Element Seed-Tensor (float32) in den niederwertigsten Bits des Bildes."""
    import numpy as np
    from PIL import Image

    # Konvertiere Bild zu numpy Array
    img_array = np.array(image)

    # Konvertiere Tensor zu numpy
    if hasattr(seed_tensor, 'cpu'):
        seed_array = seed_tensor.cpu().data.numpy()
    else:
        seed_array = seed_tensor

    # Float32 → Bytes → Binärstring
    seed_bytes = seed_array.astype(np.float32).tobytes()
    seed_binary = ''.join(f'{byte:08b}' for byte in seed_bytes)

    # Prüfe ob genug Platz im Bild ist
    flat_img = img_array.flatten()
    if len(seed_binary) > len(flat_img):
        raise ValueError("Bild zu klein zum Einbetten des Seeds.")

    # Bits in niederwertigste Bits einfügen
    for i, bit in enumerate(seed_binary):
        flat_img[i] = (flat_img[i] & 0xFE) | int(bit)

    # Zurück in Bildform
    result_array = flat_img.reshape(img_array.shape)
    return Image.fromarray(result_array.astype(np.uint8))

    

def extract_seed_from_image(image):
    """Extrahiert den 512-Element Float32 Seed-Tensor aus dem Bild."""
    import numpy as np
    import torch

    try:
        # Bilddaten lesen
        img_array = np.array(image)
        flat_img = img_array.flatten()

        # Anzahl der zu extrahierenden Bits: 512 float32 * 4 Bytes * 8 Bits = 16384
        total_bits = 512 * 4 * 8

        if len(flat_img) < total_bits:
            raise ValueError("Bild enthält nicht genug Daten für Seed.")

        # LSBs extrahieren
        binary_data = ''.join(str(pixel & 1) for pixel in flat_img[:total_bits])

        # Binär → Bytes
        seed_bytes = bytearray(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))

        # Bytes → Float32 Numpy-Array → Tensor
        seed_array = np.frombuffer(seed_bytes, dtype=np.float32)

        if seed_array.size != 512:
            raise ValueError(f"Extrahierter Seed hat falsche Größe: {seed_array.size} statt 512")

        return torch.from_numpy(seed_array)

    except Exception as e:
        print(f"Fehler beim Extrahieren: {e}")
        return None
    

class GANCardGUI:
    def __init__(self)->None:
        self.current_image = None
        self.current_seed = None
        self.setup_window()
        return None


    def setup_window(self)->None:
        self.window = Tk()
        self.window.geometry("840x840")
        self.window.configure(bg="#213055")
        self.window.resizable(False, False)

        self.setup_banner()

        self.setup_tabs()

        self.window.mainloop()
        return None


    def setup_banner(self)->None:
        self.canvas = Canvas(
            self.window,
            bg = "#213055",
            height = 840,
            width = 840,
            bd = 0,
            highlightthickness = 0,
            relief = 'ridge'
        )

        self.canvas.place(x=0, y=0)

        self.button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        self.button_1 = Button(
            image = self.button_image_1,
            borderwidth= 0,
            highlightthickness=0,
            command = lambda: webbrowser.open('https://labs.onb.ac.at/en/datasets/akon/'),
            relief='flat'
        )
        self.button_1.place(x=0.0,
                                  y=0.0,
                                  width=840.0,
                                  height=333.0
                                  )
        
        self.canvas.create_text(
            255.0,
            340.0,
            anchor="nw",
            text="GAN-CARD",
            fill="#FFFFFF",
            font=("UnifrakturCook Bold", 64*-1)
        )
        return None
    

    def setup_tabs(self)->None:
        tab_frame = Frame(self.window, bg="#213055")
        tab_frame.place(x=0,
                        y=380,
                        width=840,
                        height=460
                        )
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#213055')
        style.configure('TNotebook.Tab', background='#4E897D', foreground='white')

        self.notebook = ttk.Notebook(tab_frame)
        self.notebook.place(x=10,
                            y=10,
                            width=820,
                            height=440
                            )
        
        self.random_frame = Frame(self.notebook, bg='#213055')
        self.notebook.add(self.random_frame, text='Random')
        self.setup_random_tab()

        self.static_frame = Frame(self.notebook, bg='#213055')
        self.notebook.add(self.static_frame, text='Statisch')
        self.setup_static_tab()

        return None
    

    def setup_random_tab(self)->None:
        control_frame = Frame(self.random_frame, bg='#213055')
        control_frame.place(x=0,
                            y=0,
                            width=400,
                            height=410
                            )
        
        self.random_generate_btn = Button(
            control_frame,
            text="Generate Random",
            bg='#4E897D',
            fg='white',
            font=('Roboto Medium', 14),
            command=self.generate_random_image
        )
        self.random_generate_btn.place(x=50,
                                       y=50,
                                       width=200,
                                       height=50
                                       )
        
        info_label = Label(
            control_frame,
            text='Generiert ein zufälliges Bild\nmit verstecktem Seed',
            bg='#213055',
            fg='white',
            font = ("Roboto", 10)
        )
        info_label.place(x=50, y=120)

        self.setup_image_display(self.random_frame, "random")
        return None
    

    def setup_static_tab(self)->None:
        control_frame = Frame(self.static_frame, bg='#213055')
        control_frame.place(x=0,
                            y=0,
                            width=400,
                            height=410
                            )
        
        seed_label = Label(
            control_frame,
            text='Seed:',
            bg="#213055",
            fg='white',
            font=("Roboto Medium", 14)
        )
        seed_label.place(x=50, y=50)

        self.static_load_btn = Button(
            control_frame,
            text="Bild laden & reproduzieren",
            bg='#4E897D',
            fg='white',
            font=("Roboto Medium", 14),
            command=self.generate_static_image
        )
        self.static_load_btn.place(x=50,
                                   y=90,
                                   width=250,
                                   height=50
                                   )
        info_text = Label(
            control_frame,
            text="Lädt ein Bild mit verstecktem Seed,\nextrahiert den Seed und generiert\ndasselbe Bild neu.",
            bg="#213055",
            fg="white",
            font=("Roboto", 10),
            justify="left"
        )
        info_text.place(x=50, y=160)

        self.setup_image_display(self.static_frame, "static")
        return None
    

    def setup_image_display(self,
                            parent,
                            mode='random'
                            )->None:
        image_frame = Frame(parent, bg='#213055')
        image_frame.place(x=400,
                          y=0,
                          width=400,
                          height=410
                          )
        
        label = Label(
            image_frame,
            text="Kein Bild generiert",
            bg='#213055',
            fg='white',
            font=("Roboto", 12)
        )
        label.place(x=50,
                    y=50,
                    width=300,
                    height=250
                    )
        
        if mode == "static":
            self.static_image_label = label
        elif mode == "random":
            self.random_image_label = label

        
        self.save_btn = Button(
            image_frame,
            text="Speichern",
            bg='#4E897D',
            fg='white',
            font=("Roboto Medium", 12),
            command=self.save_image
        )
        self.save_btn.place(x=150,
                            y=320,
                            width=100,
                            height=40
                            )
        
        self.help_btn = Button(
            image_frame,
            text = "Hilfe",
            bg='#4E897D',
            fg='white',
            font=("Roboto Medium", 12),
            command=self.show_help
        )
        self.help_btn.place(x=50,
                            y=320,
                            width=80,
                            height=40
                            )
        
        self.credits_btn = Button(
            image_frame,
            text="Credits",
            bg="#4E897D",
            fg="white",
            font=("Roboto Medium", 12),
            command = self.show_credits
        )
        self.credits_btn.place(x=270,
                               y=320,
                               width=80,
                               height=40
                               )
        return None
    

    def generate_random_image(self)->None:
        try:
            random_seed = seed = torch.randn(1, 512, device=device)
            seed, image = generator.main(random_seed)

            self.current_image = hide_seed_tensor_in_image(image, seed)
            self.current_image.show()
            print("Bildgröße:", self.current_image.size)
            
            print("Bildmodus:", self.current_image.mode)
            self.current_seed = seed
            #print(image)

            self.display_image(self.current_image,
                               self.random_image_label
                               )
        except Exception as e:
            tkinter.messagebox.showerror("Fehler", f"Fehler beim Generieren: {str(e)}")
        return None
    
    
    def generate_static_image(self)->None:
        try:
            # Lade ein Bild und extrahiere den Seed
            file_path = filedialog.askopenfilename(
                title="Bild mit verstecktem Seed auswählen",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
            )
            
            if not file_path:
                return
            
            # Lade das Bild
            loaded_image = Image.open(file_path)
            
            # Extrahiere den Seed-Tensor
            extracted_seed = extract_seed_from_image(loaded_image)
            extracted_seed = extracted_seed.unsqueeze(0)
            extracted_seed = extracted_seed.view(1, -1).to(device)
            if extracted_seed is None:
                tkinter.messagebox.showerror("Fehler", "Kein gültiger 512-Element Seed im Bild gefunden!")
                return
            
            # Generiere neues Bild mit dem extrahierten Seed
            # Hier würdest du deinen generator mit dem extrahierten Seed aufrufen
            _, new_image = generator.main(extracted_seed)  # Du musst generator.main() anpassen um Seed zu akzeptieren
            
            # Verstecke den extrahierten Seed im neuen Bild
            self.current_image = hide_seed_tensor_in_image(new_image, extracted_seed)
            self.current_seed = extracted_seed
            
            # Zeige neues Bild an
            self.display_image(self.current_image,
                               self.static_image_label
                               )
            
            # Zeige Seed-Info an
            #seed_preview = f"Extrahierter Seed (ersten 5 Werte): {extracted_seed[:5].tolist()}"
            #self.seed_info_label.configure(text=seed_preview)
            
            tkinter.messagebox.showinfo("Erfolg", "Seed erfolgreich extrahiert und Bild reproduziert!")
            
        except Exception as e:
            tkinter.messagebox.showerror("Fehler", f"Fehler beim Reproduzieren: {str(e)}")
            print(f"Debug: Fehler-Details: {e}")  # Für Debugging
  

    def display_image(self,
                      image,
                      target_label
                      )->None:
        try:
            display_image = image.resize((300, 350),
                                         Image.Resampling.LANCZOS
                                        )
            self.photo = ImageTk.PhotoImage(display_image)
            target_label.configure(image=self.photo)
            target_label.image = self.photo
        except Exception as e:
            tkinter.messagebox.showerror("Fehler", f"Fehler bei Bildanzeige: {str(e)}")
        return None
    

    def load_and_extract_seed(self)->None:
        try:
            file_path = filedialog.askopenfilename(
                title="Bild auswählen", 
                filetypes=[("Image files", "*.png, *.jpg, *.jpeg, *.gif, *bmp")]
            )

            if file_path:
                image = Image.open(file_path)
                seed = extract_seed_from_image(image)

                if seed:
                    self.extracted_seed_label.configure(text=f"Extrahierter Seed: {seed}")
                    self.current_image = image
                    self.current_seed = seed
                    #self.display_image(image)

                    self.seed_entry.delete(0, 'end')
                    self.seed_entry.insert(0, seed)
                else:
                    tkinter.messagebox.showwarning("Warnung", "Kein versteckter Seed gefunden!")
        except Exception as e:
            tkinter.messagebox.showwarning("Fehler", f"Fehler beim Laden des Bildes: {str(e)}")
        return None
    

    def save_image(self)->None:
        if self.current_image is None:
            tkinter.messagebox.showwarning("Warnung", "Kein Bild zum Speichern vorhanden")
            return None
        
        try:
            file_path = filedialog.asksaveasfilename(
                title = "Bild speichern", 
                defaultextension="*.png",
                filetypes=[("PNG files", '*.png'), ('JPEG files', '*.jpeg')]
            )

            if file_path:
                self.current_image.save(file_path)
                tkinter.messagebox.showinfo("Erfolg", "Bild erfolgreich gespeichert!")
        except Exception as e:
            tkinter.messagebox.showerror("Fehler", f"Fehler beim Speichern: {str(e)}")
        return None
    

    def show_help(self)->None:
        help_text = """GAN-CARD Hilfe:
        
Random Tab:
- Generiert zufällige Bilder mit verstecktem Seed
- Laden Sie Bilder, um versteckte Seeds zu extrahieren

Statisch Tab:
- Geben Sie einen spezifischen Seed ein
- Generiert reproduzierbare Bilder

Steganografie
- Seeds werden unsichtbar in Bildern versteckt
- Laden Sie generierte Bilder, um den Seed zu extrahieren"""

        tkinter.messagebox.showinfo("Hilfe", help_text)
        return None
    

    def show_credits(self)->None:
        tkinter.messagebox.showinfo(
            'Credits',
            '             GAN-CARD v2.0\n Dennis Becker & Lukas Laaser\nEngineering ML-based Systems\n\nMit Steganografie-Feature'
        )
        return None
    

def main()->None:
    app = GANCardGUI()
    return None


if __name__ == '__main__':
    main()