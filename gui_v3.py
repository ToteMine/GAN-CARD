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
import requests
from urllib.parse import quote

from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Label, Frame, StringVar, Radiobutton
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont


device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = os.path.join(OUTPUT_PATH, 'assets/frame0/')

def relative_to_assets(path: str)->Path:
    print(ASSETS_PATH / Path(path))
    return ASSETS_PATH / Path(path)


def download_google_font(font_name="Old Standard TT", font_weight="400"):
    """
    Lädt eine Google Font herunter und speichert sie lokal.
    """
    font_dir = os.path.join(OUTPUT_PATH, 'fonts')
    os.makedirs(font_dir, exist_ok=True)
    
    font_file = os.path.join(font_dir, f"{font_name.replace(' ', '_')}.ttf")
    
    # Prüfe ob Font bereits existiert
    if os.path.exists(font_file):
        return font_file
    
    try:
        # Google Fonts API URL
        api_url = f"https://fonts.googleapis.com/css2?family={quote(font_name)}:wght@{font_weight}"
        
        # Hole CSS mit Font-URLs
        response = requests.get(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        css_content = response.text
        
        # Extrahiere die .ttf URL (vereinfachte Methode)
        import re
        ttf_urls = re.findall(r'url\((https://[^)]+\.ttf)\)', css_content)
        
        if ttf_urls:
            # Lade die erste gefundene .ttf Datei
            font_response = requests.get(ttf_urls[0])
            with open(font_file, 'wb') as f:
                f.write(font_response.content)
            return font_file
        else:
            print(f"Keine TTF-URL für {font_name} gefunden")
            return None
            
    except Exception as e:
        print(f"Fehler beim Laden der Font {font_name}: {e}")
        return None


def add_text_to_image(image, text, position="top-left", font_path=None):
    """
    Fügt Text zu einem Bild hinzu mit dynamischer Schriftgröße und Positionierung.
    
    Args:
        image: PIL Image
        text: String - der hinzuzufügende Text
        position: String - "top-left", "top-right", "bottom-left", "bottom-right"
        font_path: String - Pfad zur Font-Datei
    """
    if not text.strip():
        return image
    
    # Erstelle eine Kopie des Bildes
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Bildabmessungen
    img_width, img_height = image.size
    
    # Font laden
    if font_path and os.path.exists(font_path):
        # Dynamische Schriftgröße basierend auf Textlänge und Bildgröße
        base_font_size = max(20, min(img_width // 15, img_height // 20))
        text_length_factor = max(0.5, 1.0 - (len(text) - 10) * 0.02)
        font_size = int(base_font_size * text_length_factor)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
    else:
        # Fallback auf Default Font
        font = ImageFont.load_default()
    
    # Textabmessungen berechnen
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Margin vom Rand
    margin = 20
    
    # Position berechnen
    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x, y = img_width - text_width - margin, margin
    elif position == "bottom-left":
        x, y = margin, img_height - text_height - margin
    elif position == "bottom-right":
        x, y = img_width - text_width - margin, img_height - text_height - margin
    else:
        x, y = margin, margin  # Default
    
    # Text mit Schatten für bessere Lesbarkeit
    shadow_offset = 2
    # Schatten
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill="black")
    # Haupttext
    draw.text((x, y), text, font=font, fill="white")
    
    return img_with_text


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
        self.font_path = None
        self.load_font()
        self.setup_window()
        return None

    def load_font(self):
        """Lädt die Google Font beim Start"""
        print("Lade Google Font...")
        self.font_path = download_google_font("Old Standard TT", "400")
        if self.font_path:
            print(f"Font erfolgreich geladen: {self.font_path}")
        else:
            print("Font konnte nicht geladen werden, verwende Standard-Font")

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
                                       y=20,
                                       width=200,
                                       height=50
                                       )
        
        # Grußformel-Eingabe
        greeting_label = Label(
            control_frame,
            text='Grußformel:',
            bg='#213055',
            fg='white',
            font=('Roboto Medium', 12)
        )
        greeting_label.place(x=50, y=80)
        
        self.random_greeting_entry = Entry(
            control_frame,
            font=('Roboto', 10),
            width=30
        )
        self.random_greeting_entry.place(x=50, y=105)
        
        # Position auswählen
        position_label = Label(
            control_frame,
            text='Position:',
            bg='#213055',
            fg='white',
            font=('Roboto Medium', 12)
        )
        position_label.place(x=50, y=140)
        
        self.random_position_var = StringVar(value="top-left")
        
        # Radiobuttons in Quadrat-Anordnung
        positions = [
            ("top-left", "↖", 160, 10),
            ("top-right", "↗", 160, 60),
            ("bottom-left", "↙", 190, 10),
            ("bottom-right", "↘", 190, 60)
        ]
        
        for value, symbol, y, x_offset in positions:
            rb = Radiobutton(
                control_frame,
                text=symbol,
                variable=self.random_position_var,
                value=value,
                bg='#213055',
                fg='white',
                selectcolor='#4E897D',
                font=('Roboto', 16),
                indicatoron=0,
                width=3,
                height=1
            )
            rb.place(x=50 + x_offset, y=y)
        
        info_label = Label(
            control_frame,
            text='Generiert ein zufälliges Bild\nmit verstecktem Seed',
            bg='#213055',
            fg='white',
            font = ("Roboto", 10)
        )
        info_label.place(x=50, y=230)

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
        seed_label.place(x=50, y=20)

        self.static_load_btn = Button(
            control_frame,
            text="Bild laden & reproduzieren",
            bg='#4E897D',
            fg='white',
            font=("Roboto Medium", 14),
            command=self.generate_static_image
        )
        self.static_load_btn.place(x=50,
                                   y=50,
                                   width=250,
                                   height=40
                                   )
        
        # Grußformel-Eingabe
        greeting_label = Label(
            control_frame,
            text='Grußformel:',
            bg='#213055',
            fg='white',
            font=('Roboto Medium', 12)
        )
        greeting_label.place(x=50, y=100)
        
        self.static_greeting_entry = Entry(
            control_frame,
            font=('Roboto', 10),
            width=30
        )
        self.static_greeting_entry.place(x=50, y=125)
        
        # Position auswählen
        position_label = Label(
            control_frame,
            text='Position:',
            bg='#213055',
            fg='white',
            font=('Roboto Medium', 12)
        )
        position_label.place(x=50, y=155)
        
        self.static_position_var = StringVar(value="top-left")
        
        # Radiobuttons in Quadrat-Anordnung
        positions = [
            ("top-left", "↖", 175, 10),
            ("top-right", "↗", 175, 60),
            ("bottom-left", "↙", 205, 10),
            ("bottom-right", "↘", 205, 60)
        ]
        
        for value, symbol, y, x_offset in positions:
            rb = Radiobutton(
                control_frame,
                text=symbol,
                variable=self.static_position_var,
                value=value,
                bg='#213055',
                fg='white',
                selectcolor='#4E897D',
                font=('Roboto', 16),
                indicatoron=0,
                width=3,
                height=1
            )
            rb.place(x=50 + x_offset, y=y)
        
        info_text = Label(
            control_frame,
            text="Lädt ein Bild mit verstecktem Seed,\nextrahiert den Seed und generiert\ndasselbe Bild neu.",
            bg="#213055",
            fg="white",
            font=("Roboto", 10),
            justify="left"
        )
        info_text.place(x=50, y=240)

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
            
            # Grußformel hinzufügen VOR der Steganographie
            greeting_text = self.random_greeting_entry.get()
            position = self.random_position_var.get()
            
            if greeting_text.strip():
                image = add_text_to_image(image, greeting_text, position, self.font_path)

            self.current_image = hide_seed_tensor_in_image(image, seed)
            self.current_seed = seed

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
            _, new_image = generator.main(extracted_seed)
            
            # Grußformel hinzufügen VOR der Steganographie
            greeting_text = self.static_greeting_entry.get()
            position = self.static_position_var.get()
            
            if greeting_text.strip():
                new_image = add_text_to_image(new_image, greeting_text, position, self.font_path)
            
            # Verstecke den extrahierten Seed im neuen Bild
            self.current_image = hide_seed_tensor_in_image(new_image, extracted_seed)
            self.current_seed = extracted_seed
            
            # Zeige neues Bild an
            self.display_image(self.current_image,
                               self.static_image_label
                               )
            
            tkinter.messagebox.showinfo("Erfolg", "Seed erfolgreich extrahiert und Bild reproduziert!")
            
        except Exception as e:
            tkinter.messagebox.showerror("Fehler", f"Fehler beim Reproduzieren: {str(e)}")
            print(f"Debug: Fehler-Details: {e}")
  

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
- Grußformel optional hinzufügen
- Position der Grußformel wählen (↖↗↙↘)

Statisch Tab:
- Lädt Bilder mit verstecktem Seed
- Reproduziert das Bild mit neuem Text
- Grußformel optional hinzufügen

Text-Features:
- Dynamische Schriftgröße je nach Textlänge
- Schrift mit Schatten für bessere Lesbarkeit
- Google Font "Old Standard TT" wird verwendet

Steganografie:
- Seeds werden unsichtbar in Bildern versteckt
- Laden Sie generierte Bilder, um den Seed zu extrahieren"""

        tkinter.messagebox.showinfo("Hilfe", help_text)
        return None
    

    def show_credits(self)->None:
        tkinter.messagebox.showinfo(
            'Credits',
            '             GAN-CARD v2.1\n Dennis Becker & Lukas Laaser\nEngineering ML-based Systems\n\nMit Steganografie & Grußformel-Feature\nGoogle Font: Old Standard TT'
        )
        return None
    

def main()->None:
    app = GANCardGUI()
    return None


if __name__ == '__main__':
    main()