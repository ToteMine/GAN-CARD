import webbrowser
import numpy as np
import os

import requests
import tempfile
import shutil
import gc
import json
import threading
import generator
import re
import tkinter as tk

from urllib.parse import quote
from pathlib import Path
from tkinter import ttk, filedialog, Tk, Canvas, Entry, Button, PhotoImage, Label, Frame, StringVar, Radiobutton, \
    messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = os.path.join(OUTPUT_PATH, 'assets/frame0/')


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


"""
Lädt eine Google Font herunter und speichert sie lokal.
"""


def download_google_font(font_name="Old Standard TT",
                         font_weight="400"
                         ) -> str:
    font_dir = os.path.join(OUTPUT_PATH,
                            'fonts'
                            )
    os.makedirs(font_dir,
                exist_ok=True
                )

    font_file = os.path.join(font_dir,
                             f"{font_name.replace(' ', '_')}.ttf"
                             )

    if os.path.exists(font_file):
        return font_file

    try:
        api_url = f"https://fonts.googleapis.com/css2?family={quote(font_name)}:wght@{font_weight}"

        response = requests.get(api_url,
                                headers={'User-Agent': 'Mozilla/5.0'}
                                )
        css_content = response.text
        ttf_urls = re.findall(r'url\((https://[^)]+\.ttf)\)',
                              css_content
                              )

        if ttf_urls:
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


"""
Fügt Text zu einem Bild hinzu mit dynamischer Schriftgröße und Positionierung.

Args:
    image: PIL Image
    text: String - der hinzuzufügende Text
    position: String - "top-left", "top-right", "bottom-left", "bottom-right"
    font_path: String - Pfad zur Font-Datei
"""


def add_text_to_image(image: object,
                      text: str,
                      position="top-left",
                      font_path=None
                      ) -> Image:
    if not text.strip():
        return image

    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)

    img_width, img_height = image.size

    if font_path and os.path.exists(font_path):
        base_font_size = max(20, min(img_width // 15, img_height // 20))
        text_length_factor = max(0.5, 1.0 - (len(text) - 10) * 0.02)
        font_size = int(base_font_size * text_length_factor)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = 20

    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x, y = img_width - text_width - margin, margin
    elif position == "bottom-left":
        x, y = margin, img_height - text_height - margin
    elif position == "bottom-right":
        x, y = img_width - text_width - margin, img_height - text_height - margin
    else:
        x, y = margin, margin

    shadow_offset = 2
    draw.text((x + shadow_offset, y + shadow_offset),
              text,
              font=font,
              fill="black"
              )
    draw.text((x, y),
              text,
              font=font,
              fill="white"
              )
    return img_with_text


"""
Versteckt den 512-Element Seed-Tensor (float32) in den niederwertigsten Bits des Bildes.
"""


def hide_seed_tensor_in_image(image: object,
                              seed_tensor: object
                              ) -> Image:
    img_array = np.array(image)
    if hasattr(seed_tensor, 'cpu'):
        seed_array = seed_tensor.cpu().data.numpy()
    else:
        seed_array = seed_tensor

    seed_bytes = seed_array.astype(np.float32).tobytes()
    seed_binary = ''.join(f'{byte:08b}' for byte in seed_bytes)

    flat_img = img_array.flatten()
    if len(seed_binary) > len(flat_img):
        raise ValueError("Bild zu klein zum Einbetten des Seeds.")

    for i, bit in enumerate(seed_binary):
        flat_img[i] = (flat_img[i] & 0xFE) | int(bit)

    result_array = flat_img.reshape(img_array.shape)
    return Image.fromarray(result_array.astype(np.uint8))


"""
Extrahiert den 512-Element Float32 Seed-Tensor aus dem Bild.
"""


def extract_seed_from_image(image: object) -> object:
    try:
        img_array = np.array(image)
        flat_img = img_array.flatten()
        total_bits = 512 * 4 * 8

        if len(flat_img) < total_bits:
            raise ValueError("Bild enthält nicht genug Daten für Seed.")
        binary_data = ''.join(str(pixel & 1) for pixel in flat_img[:total_bits])
        seed_bytes = bytearray(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8))
        seed_array = np.frombuffer(seed_bytes, dtype=np.float32)
        if seed_array.size != 512:
            raise ValueError(f"Extrahierter Seed hat falsche Größe: {seed_array.size} statt 512")
        return seed_array
    except Exception as e:
        print(f"Fehler beim Extrahieren: {e}")
        return None


"""
Speichert die Metadaten zur Bild- und Gifgenerierung in einer JSON File.
Besonders wichtig für die Reproduktion der GIFs
"""


def save_metadata_to_json(path: str,
                          seed_tensor: object,
                          n_steps=None,
                          n_keypoints=None,
                          fps=None,
                          keypoints=None
                          ) -> None:
    data = {
        "seed": seed_tensor.squeeze().tolist(),
        "n_steps": n_steps,
        "n_keypoints": n_keypoints,
        "fps": fps,
        "keypoints": [kp.squeeze().tolist() for kp in keypoints] if keypoints is not None else None
    }
    with open(path, 'w') as f:
        json.dump(data,
                  f,
                  indent=2
                  )


"""
Daten zur Reproduktion werden aus einer angegebenen JSON-File geladen.
"""


def load_metadata_from_json(json_path: str) -> list:
    with open(json_path, 'r') as f:
        data = json.load(f)

    seed_array = None
    if data.get("seed") is not None:
        seed_array = np.array(data["seed"], dtype=np.float32).reshape(1, -1)

    keypoints = None
    if data.get("keypoints") is not None:
        keypoints = [np.array(kp, dtype=np.float32).reshape(1, -1) for kp in data["keypoints"]]

    return seed_array, data.get("n_steps"), data.get("n_keypoints"), data.get("fps"), keypoints


"""
Klasse für die GUI
"""


class GANCardGUI:
    """
    Klassen Initiator
    """

    def __init__(self) -> None:
        self.current_image = None
        self.current_seed = None
        self.current_n_steps = None
        self.current_n_keypoints = None
        self.current_keypoints = None
        self.current_fps = None
        self.font_path = None
        self.random_gif_after_id = None
        self.static_gif_after_id = None
        self.load_font()
        self.setup_window()
        self.animation_frames = []
        self.is_gif_mode = False
        return None

    """Lädt die Google Font beim Start"""

    def load_font(self) -> None:
        print("Lade Google Font...")
        self.font_path = download_google_font("Old Standard TT", "400")
        if not self.font_path:
            messagebox.showerror("Font konnte nicht geladen werden, verwende Standard-Font")
        return None

    """
    Baut das Fenster und dessen Inhalte auf.
    """

    def setup_window(self) -> None:
        self.window = Tk()
        self.window.geometry("840x840")
        self.window.configure(bg="#213055")
        self.window.resizable(False, False)
        self.window.title("GAN-CARD")

        self.setup_banner()

        self.setup_tabs()

        self.window.mainloop()
        return None

    """
    Erstellt den Header - inkl. verstecktem Button - und
    den Titel des Programms.
    """

    def setup_banner(self) -> None:
        self.canvas = Canvas(
            self.window,
            bg="#213055",
            height=840,
            width=840,
            bd=0,
            highlightthickness=0,
            relief='ridge'
        )

        self.canvas.place(x=0, y=0)

        self.button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: webbrowser.open('https://labs.onb.ac.at/en/datasets/akon/'),
            relief='flat'
        )
        self.button_1.place(x=0.0,
                            y=0.0,
                            width=840.0,
                            height=333.0
                            )

        self.canvas.create_text(
            320.0,
            340.0,
            anchor="nw",
            text="GAN-CARD",
            fill="#FFFFFF",
            font=("UnifrakturCook Bold", 45 * -1)
        )
        return None

    """
    Baut den Random-Tab und den Static-Tab und bindet diese an das Fenster.
    """

    def setup_tabs(self) -> None:
        tab_frame = Frame(self.window, bg="#213055")
        tab_frame.place(x=0,
                        y=380,
                        width=840,
                        height=460
                        )

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook',
                        background='#213055'
                        )
        style.configure('TNotebook.Tab',
                        background='#4E897D',
                        foreground='white'
                        )

        self.notebook = ttk.Notebook(tab_frame)
        self.notebook.place(x=10,
                            y=10,
                            width=820,
                            height=440
                            )

        self.random_frame = Frame(self.notebook,
                                  bg='#213055'
                                  )
        self.notebook.add(self.random_frame,
                          text='Generieren'
                          )
        self.setup_random_tab()

        self.static_frame = Frame(self.notebook,
                                  bg='#213055'
                                  )
        self.notebook.add(self.static_frame,
                          text='Reproduzieren'
                          )
        self.setup_static_tab()
        return None

    """
    Baut den Random-Tab.
    """

    def setup_random_tab(self) -> None:
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

        self.random_animation_var = tk.BooleanVar()
        self.random_animation_checkbox = tk.Checkbutton(
            control_frame,
            text="GIF Animation\nerstellen",
            variable=self.random_animation_var,
            bg='#213055',
            fg='white',
            selectcolor='#4E897D',
            font=('Roboto', 10),
            command=self.toggle_animation_controls
        )
        self.random_animation_checkbox.place(x=260,
                                             y=35
                                             )
        self.random_anim_frame = Frame(control_frame,
                                       bg='#213055'
                                       )
        self.random_anim_frame.place(x=50,
                                     y=70,
                                     width=300,
                                     height=250
                                     )

        tk.Label(self.random_anim_frame,
                 text="Animation Steps:",
                 bg='#213055',
                 fg='white',
                 font=('Roboto', 9)
                 ).place(x=0, y=0)
        self.random_steps_var = tk.IntVar(value=60)
        self.random_steps_scale = tk.Scale(
            self.random_anim_frame,
            from_=10, to=200,
            orient='horizontal',
            variable=self.random_steps_var,
            bg='#213055', fg='white',
            highlightbackground='#4E897D',
            length=200
        )
        self.random_steps_scale.place(x=0,
                                      y=20
                                      )

        tk.Label(self.random_anim_frame,
                 text="Keypoints:",
                 bg='#213055',
                 fg='white',
                 font=('Roboto', 9)
                 ).place(x=0, y=65)
        self.random_keypoints_var = tk.IntVar(value=8)
        self.random_keypoints_scale = tk.Scale(
            self.random_anim_frame,
            from_=3, to=20,
            orient='horizontal',
            variable=self.random_keypoints_var,
            bg='#213055', fg='white',
            highlightbackground='#4E897D',
            length=200
        )
        self.random_keypoints_scale.place(x=0,
                                          y=85
                                          )

        tk.Label(self.random_anim_frame,
                 text="FPS",
                 bg='#213055',
                 fg='white',
                 font=('Roboto', 9)
                 ).place(x=0, y=130)
        self.fps_var = tk.IntVar(value=30)
        self.fps_scale = tk.Scale(
            self.random_anim_frame,
            from_=5, to=30,
            orient='horizontal',
            variable=self.fps_var,
            bg='#213055', fg='white',
            highlightbackground='#4E897D',
            length=200
        )
        self.fps_scale.place(x=0,
                             y=150
                             )
        self.random_anim_frame.place_forget()

        greeting_label = Label(
            control_frame,
            text='Grußformel:',
            bg='#213055',
            fg='white',
            font=('Roboto Medium', 12)
        )
        greeting_label.place(x=50,
                             y=265
                             )
        self.random_greeting_entry = Entry(
            control_frame,
            font=('Roboto', 10),
            width=30
        )
        self.random_greeting_entry.place(x=50,
                                         y=285
                                         )
        position_label = Label(
            control_frame,
            text='Position:',
            bg='#213055',
            fg='white',
            font=('Roboto Medium', 12)
        )
        position_label.place(x=50,
                             y=305
                             )
        self.random_position_var = StringVar(value="top-left")
        positions = [
            ("top-left", "↖", 325, 10),
            ("top-right", "↗", 325, 60),
            ("bottom-left", "↙", 360, 10),
            ("bottom-right", "↘", 360, 60)
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
            rb.place(x=50 + x_offset,
                     y=y
                     )

        info_label = Label(
            control_frame,
            text='Generiert ein zufälliges Bild\nmit verstecktem Seed',
            bg='#213055',
            fg='white',
            font=("Roboto", 10)
        )
        info_label.place(x=160,
                         y=340
                         )

        self.setup_image_display(self.random_frame, "random")
        return None

    """
    Baut den Static-Tab.
    """

    def setup_static_tab(self) -> None:
        control_frame = Frame(self.static_frame, bg='#213055')
        control_frame.place(x=0,
                            y=0,
                            width=400,
                            height=410
                            )

        self.static_load_btn = Button(
            control_frame,
            text="Bild laden & reproduzieren",
            bg='#4E897D',
            fg='white',
            font=("Roboto Medium", 14),
            command=self.generate_static_image
        )
        self.static_load_btn.place(x=50,
                                   y=20,
                                   width=250,
                                   height=40
                                   )

        self.static_info_label = Label(
            control_frame,
            text="Keine Datei geladen",
            bg='#213055',
            fg='white',
            font=('Roboto', 9),
            justify='left'
        )
        self.static_info_label.place(x=50,
                                     y=70
                                     )

        greeting_label = Label(
            control_frame,
            text='Grußformel:',
            bg='#213055',
            fg='white',
            font=('Roboto Medium', 12)
        )
        greeting_label.place(x=50,
                             y=100
                             )
        self.static_greeting_entry = Entry(
            control_frame,
            font=('Roboto', 10),
            width=30
        )
        self.static_greeting_entry.place(x=50,
                                         y=120
                                         )

        position_label = Label(
            control_frame,
            text='Position:',
            bg='#213055',
            fg='white',
            font=('Roboto Medium', 12)
        )
        position_label.place(x=50,
                             y=150
                             )
        self.static_position_var = StringVar(value="top-left")
        positions = [
            ("top-left", "↖", 170, 10),
            ("top-right", "↗", 170, 60),
            ("bottom-left", "↙", 200, 10),
            ("bottom-right", "↘", 200, 60)
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
            rb.place(x=50 + x_offset,
                     y=y
                     )

        info_text = Label(
            control_frame,
            text="Lädt ein Bild mit verstecktem Seed,\nextrahiert den Seed und generiert\ndasselbe Bild neu.",
            bg="#213055",
            fg="white",
            font=("Roboto", 10),
            justify="left"
        )
        info_text.place(x=50,
                        y=250
                        )

        self.setup_image_display(self.static_frame, "static")
        return None

    """
    Baut das Image Display innerhalb der Tabs, auf welchem das generierte Bild
    oder Gif angezeigt wird.
    """

    def setup_image_display(self,
                            parent: object,
                            mode='random'
                            ) -> None:
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

        self.progressbar = None
        progressbar = ttk.Progressbar(
            image_frame,
            orient="horizontal",
            length=250,
            mode="determinate"
        )
        progressbar.place(x=75,
                          y=200,
                          width=250,
                          height=20
                          )
        # Falls die Progressbar zu Beginn nicht angezeigt werden soll.
        # progressbar.lower()

        if mode == "static":
            self.static_image_label = label
            self.static_progressbar = progressbar
        elif mode == "random":
            self.random_image_label = label
            self.random_progressbar = progressbar

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
            text="Hilfe",
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
            command=self.show_credits
        )
        self.credits_btn.place(x=270,
                               y=320,
                               width=80,
                               height=40
                               )
        return None

    """
    Fügt die Einstellungen für die Gif-Generierung beim setzen der Checkbox in
    den Random-Tab.
    """

    def toggle_animation_controls(self):
        """Zeigt/versteckt die Animation-Controls"""
        if self.random_animation_var.get():
            self.random_anim_frame.place(x=50,
                                         y=70,
                                         width=300,
                                         height=250
                                         )
        else:
            self.random_anim_frame.place_forget()

    """
    Funktion für den Generierungsbutton innerhalb des Random-Tab.
    Unterscheidet zwischen der Generierung eines Gifs und eines
    Bildes.
    """

    def generate_random_image(self) -> None:
        try:
            if self.random_animation_var.get():
                self.generate_random_gif()
            else:
                self.show_progressbar(1)
                random_seed = np.random.randn(1, 512).astype(np.float32)
                seed, image = generator.main(random_seed)

                greeting_text = self.random_greeting_entry.get()
                position = self.random_position_var.get()

                if greeting_text.strip():
                    image = add_text_to_image(image,
                                              greeting_text,
                                              position,
                                              self.font_path
                                              )

                self.current_image = hide_seed_tensor_in_image(image,
                                                               seed
                                                               )
                self.current_seed = seed
                self.is_gif_mode = False

                self.display_image(self.current_image,
                                   self.random_image_label,
                                   mode='random'
                                   )
                self.hide_progressbar()
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Generieren: {str(e)}")
        return None

    """
    Generiert die Frames für ein zufälliges GIF.
    """

    def generate_random_gif(self) -> Image:
        try:
            self.current_n_steps = self.random_steps_var.get()
            self.current_n_keypoints = self.random_keypoints_var.get()
            self.current_fps = self.fps_var.get()
            self.current_seed = np.random.randn(1, 512).astype(np.float32)

            temp_dir = tempfile.mkdtemp()
            self.show_progressbar(self.current_n_steps,
                                  "random"
                                  )

            def worker():
                frames = self.generate_gif_frames(self.current_n_steps,
                                                  self.current_n_keypoints,
                                                  progress_callback=self.update_progressbar
                                                  )
                self.after_gif_generation(frames,
                                          temp_dir
                                          )

            threading.Thread(target=worker,
                             daemon=True
                             ).start()
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim GIF generieren: {str(e)}")

    """
    Nimmt die zufällig generierten Bilder für ein GIF und bastelt es daraus.
    Lässt es am Ende auf dem Random-Tab im Image-Display abspielen.
    """

    def after_gif_generation(self,
                             frames: list,
                             temp_dir: str
                             ) -> None:
        greeting_text = self.random_greeting_entry.get()
        position = self.random_position_var.get()
        if greeting_text.strip():
            frames = [add_text_to_image(frame,
                                        greeting_text,
                                        position,
                                        self.font_path
                                        )
                      for frame in frames]
        gif_path = os.path.join(temp_dir, "animation.gif")
        frames[0].save(gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       duration=int(1000 / self.current_fps),
                       loop=0
                       )
        im = Image.open(gif_path)
        self.current_image = im.copy()
        im.close()
        self.animation_frames = frames
        self.is_gif_mode = True
        delay_ms = int(1000 / self.current_fps)
        self.play_gif(frames,
                      self.random_image_label,
                      delay_ms,
                      mode='random'
                      )
        shutil.rmtree(temp_dir)
        self.hide_progressbar()
        return None

    """
    Nimmt die zufällig generierten Bilder für ein GIF und bastelt es daraus.
    Lässt es am Ende auf dem Static-Tab im Image-Display abspielen.
    """

    def after_gif_generation_static(self,
                                    frames: list
                                    ) -> None:
        greeting_text = self.static_greeting_entry.get()
        position = self.static_position_var.get()
        if greeting_text.strip():
            frames = [add_text_to_image(f, greeting_text, position, self.font_path) for f in frames]
        temp_dir = tempfile.mkdtemp()
        gif_path = os.path.join(temp_dir, "animation.gif")
        frames[0].save(gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       duration=int(1000 / self.current_fps),
                       loop=0
                       )
        im = Image.open(gif_path)
        self.current_image = im.copy()
        im.close()
        self.animation_frames = frames
        self.is_gif_mode = True
        delay_ms = int(1000 / self.current_fps)
        self.play_gif(frames,
                      self.static_image_label,
                      delay_ms,
                      mode='static'
                      )
        shutil.rmtree(temp_dir)
        self.hide_progressbar()
        return None

    """
    Erzeugt zufällige GIF-Images.
    """

    def generate_gif_frames(self,
                            n_steps: int,
                            n_keypoints: int,
                            progress_callback=None
                            ) -> list:
        frames = []
        keypoints = []
        for i in range(n_keypoints):
            keypoints.append(np.random.randn(1, 512).astype(np.float32))
        self.current_keypoints = keypoints
        self.current_seed = keypoints[0]

        for step in range(n_steps):
            t = (step / n_steps) * n_keypoints
            current_idx = int(t) % n_keypoints
            next_idx = (current_idx + 1) % n_keypoints
            alpha = t - int(t)

            z_interpolated = (1 - alpha) * keypoints[current_idx] + alpha * keypoints[next_idx]

            _, image = generator.main(z_interpolated)
            frames.append(image)

            del z_interpolated
            del image
            gc.collect()
            if progress_callback is not None:
                progress_callback(step + 1)
        return frames

    """
    Nimmt die aus der JSON-File ausgelesenen keypoints und generiert die Bilder damit,
    um ein GIF zu reproduzieren.
    """

    def generate_gif_frames_from_keypoints(self,
                                           keypoints: list,
                                           n_steps: int,
                                           progress_callback=None
                                           ) -> list:
        frames = []
        n_keypoints = len(keypoints)
        for step in range(n_steps):
            t = (step / n_steps) * n_keypoints
            current_idx = int(t) % n_keypoints
            next_idx = (current_idx + 1) % n_keypoints
            alpha = t - int(t)
            z_interpolated = (1 - alpha) * keypoints[current_idx] + alpha * keypoints[next_idx]
            _, image = generator.main(z_interpolated)
            frames.append(image)
            del z_interpolated, image
            gc.collect()
            if progress_callback is not None:
                progress_callback(step + 1)
        return frames

    """
    Funktion hinter dem Generierungsbutton auf dem Statischen-Tab.
    Kann folgende Fälle unterscheiden:
        Ein Bild wird eingelesen --> Der Seed wird aus dem Bild ausgelesen, um damit ein Bild neu zu generieren.
        Die zu einem Bild gehörige JSON wird gelesen --> Wie bei dem Beispiel mit dem Bild nur dass der Seed aus
            der JSON-File gelesen wird.
        Ein GIF wird eingelesen --> Die Metadaten des GIFs werden aus der JSON ausgelesen und das GIF wird reporoduziert.
        Die zu einem GIF gehörige JSON wird eingelesen --> Die Metadaten werden ausgelesen und das zu der JSON gehörige 
            GIF wird reproduziert
    """

    def generate_static_image(self) -> None:
        try:
            file_path = filedialog.askopenfilename(
                title="Bild, GIF oder JSON auswählen",
                filetypes=[
                    ("Bilddateien", "*.png *.jpg *.jpeg *.bmp *.gif *.json"),
                    ("Alle Dateien", "*.*")
                ]
            )
            if not file_path:
                return

            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == ".json":
                metadata = {}
                with open(file_path, "r") as f:
                    metadata = json.load(f)
                if metadata.get('keypoints') is not None:
                    (
                        self.current_seed, self.current_n_steps,
                        self.current_n_keypoints, self.current_fps,
                        self.current_keypoints
                    ) = load_metadata_from_json(file_path)

                    self.show_progressbar(self.current_n_steps, "static")

                    def worker():
                        frames = self.generate_gif_frames_from_keypoints(self.current_keypoints,
                                                                         self.current_n_steps,
                                                                         progress_callback=self.update_progressbar
                                                                         )
                        self.after_gif_generation_static(frames)

                    threading.Thread(target=worker,
                                     daemon=True
                                     ).start()
                elif metadata.get('keypoints') is None:
                    self.current_seed, *_ = load_metadata_from_json(file_path)
                    self.show_progressbar(1, "static")

                    def worker():
                        _, new_image = generator.main(self.current_seed)
                        greeting_text = self.static_greeting_entry.get()
                        position = self.static_position_var.get()
                        if greeting_text.strip():
                            new_image = add_text_to_image(new_image,
                                                          greeting_text,
                                                          position,
                                                          self.font_path
                                                          )
                        self.current_image = hide_seed_tensor_in_image(new_image,
                                                                       self.current_seed
                                                                       )
                        self.is_gif_mode = False
                        self.display_image(self.current_image,
                                           self.static_image_label,
                                           mode='static'
                                           )
                        self.hide_progressbar()

                    threading.Thread(target=worker,
                                     daemon=True
                                     ).start()
                else:
                    messagebox.showerror("Fehler", "Unbekannter JSON-Inhalt.")
            elif file_ext == ".gif":
                json_path = os.path.splitext(file_path)[0] + ".json"
                if not os.path.exists(json_path):
                    messagebox.showerror("Fehler", f"Keine passende JSON-Datei gefunden:\n{json_path}")
                    return
                (
                    self.current_seed, self.current_n_steps,
                    self.current_n_keypoints, self.current_fps,
                    self.current_keypoints
                ) = load_metadata_from_json(json_path)

                self.show_progressbar(self.current_n_steps, "static")

                def worker():
                    frames = self.generate_gif_frames_from_keypoints(self.current_keypoints,
                                                                     self.current_n_steps,
                                                                     progress_callback=self.update_progressbar
                                                                     )
                    self.after_gif_generation_static(frames)

                threading.Thread(target=worker,
                                 daemon=True
                                 ).start()
            elif file_ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                self.show_progressbar(1, "static")

                def worker():
                    loaded_image = Image.open(file_path)
                    extracted_seed = extract_seed_from_image(loaded_image)
                    if extracted_seed is None:
                        messagebox.showerror("Fehler", "Kein gültiger Seed gefunden.")
                        self.hide_progressbar()
                        return
                    self.current_seed = extracted_seed.reshape(1, -1)
                    _, new_image = generator.main(self.current_seed)
                    greeting_text = self.static_greeting_entry.get()
                    position = self.static_position_var.get()
                    if greeting_text.strip():
                        new_image = add_text_to_image(new_image,
                                                      greeting_text,
                                                      position,
                                                      self.font_path
                                                      )
                    self.current_image = hide_seed_tensor_in_image(new_image,
                                                                   self.current_seed
                                                                   )
                    self.is_gif_mode = False
                    self.display_image(self.current_image,
                                       self.static_image_label,
                                       mode='static'
                                       )
                    self.hide_progressbar()

                threading.Thread(target=worker,
                                 daemon=True
                                 ).start()
            else:
                messagebox.showerror("Fehler", "Nicht unterstütztes Dateiformat.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Reproduzieren: {str(e)}")
        return None

    """
    Macht die Progressbar innerhalb des jeweiligen Tabs sichtbar.
    """

    def show_progressbar(self,
                         max_steps: int,
                         mode="random"
                         ) -> None:
        if mode == "random":
            self.progressbar = self.random_progressbar
        else:
            self.progressbar = self.static_progressbar
        self.progressbar["maximum"] = max_steps
        self.progressbar["value"] = 0
        self.progressbar.lift()
        self.progressbar.update()
        return None

    """
    Update für die Progressbar.
    """

    def update_progressbar(self,
                           value: int
                           ) -> None:
        if self.progressbar:
            self.progressbar["value"] = value
            self.progressbar.update()
        return None

    """
    Befehl zu verstecken der Progressbar.
    """

    def hide_progressbar(self):
        if self.progressbar:
            self.progressbar.lower()
        return None

    """
    Spielt das GIF innerhalb des Image-Displays des Tabs ab.
    """

    def play_gif(self,
                 frames: list,
                 label: object,
                 delay_ms=100,
                 index=0,
                 mode='random'
                 ) -> None:
        after_id_attr = f'{mode}_gif_after_id'
        if index == 0:
            prev_id = getattr(self,
                              after_id_attr,
                              None
                              )
            if prev_id:
                self.window.after_cancel(prev_id)
                setattr(self,
                        after_id_attr,
                        None
                        )

        if not frames:
            return None

        display_image = self.pad_center_image(frames[index % len(frames)])
        photo = ImageTk.PhotoImage(display_image)
        label.configure(image=photo)
        label.image = photo

        after_id = self.window.after(delay_ms,
                                     self.play_gif,
                                     frames,
                                     label,
                                     delay_ms,
                                     (index + 1) % len(frames),
                                     mode
                                     )
        setattr(self,
                after_id_attr,
                after_id
                )
        return None

    """
    Bereitet das Bild - inkl. Hintergrund - für die Anzeige vor.
    """

    def pad_center_image(self,
                         img: object,
                         target_size=(300, 350),
                         bg_color=(33, 48, 85)
                         ) -> Image:
        w, h = img.size
        tw, th = target_size
        bg = Image.new('RGB',
                       (tw, th),
                       bg_color
                       )
        x = (tw - w) // 2
        y = (th - h) // 2
        bg.paste(img,
                 (x, y)
                 )
        return bg

    """
    Anzeige des Bildes, ggf. muss die vorherige Animation im Tab abgebrochen werden.
    """

    def display_image(self,
                      image: object,
                      target_label: object,
                      mode='random'
                      ) -> None:
        after_id_attr = f'{mode}_gif_after_id'
        prev_id = getattr(self,
                          after_id_attr,
                          None
                          )
        if prev_id is not None:
            self.window.after_cancel(prev_id)
            setattr(self,
                    after_id_attr,
                    None
                    )
        try:
            display_image = self.pad_center_image(image)
            self.photo = ImageTk.PhotoImage(display_image)
            target_label.configure(image=self.photo)
            target_label.image = self.photo
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei Bildanzeige: {str(e)}")
        return None

    """
    Funktion hinter dem Speichern-Button. Erstellt einen Ordner für das jeweilige Bild
    oder GIF und speichert neben diesen noch die JSON-File mit den Metadaten innerhalb
    des Ordners.
    """

    def save_image(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("Warnung", "Kein Bild zum Speichern vorhanden")
            return
        try:
            if self.is_gif_mode:
                file_path = filedialog.asksaveasfilename(
                    title="GIF speichern",
                    defaultextension=".gif",
                    filetypes=[("GIF files", "*.gif"), ("PNG files", "*.png")]
                )
                if file_path:
                    if file_path.endswith('.gif'):
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        parent_dir = os.path.dirname(file_path)
                        save_dir = os.path.join(parent_dir,
                                                base_name
                                                )

                        os.makedirs(save_dir,
                                    exist_ok=True
                                    )

                        gif_path = os.path.join(save_dir,
                                                f"{base_name}.gif"
                                                )
                        self.animation_frames[0].save(
                            gif_path,
                            save_all=True,
                            append_images=self.animation_frames[1:],
                            duration=int(1000 / self.current_fps),
                            loop=0
                        )

                        json_path = os.path.join(save_dir, f"{base_name}.json")
                        save_metadata_to_json(
                            json_path,
                            seed_tensor=self.current_seed,
                            n_steps=self.current_n_steps,
                            n_keypoints=self.current_n_keypoints,
                            fps=self.current_fps,
                            keypoints=self.current_keypoints
                        )
                    else:
                        self.animation_frames[0].save(file_path)
                    messagebox.showinfo("Erfolg", "GIF erfolgreich gespeichert!")
            else:
                file_path = filedialog.asksaveasfilename(
                    title="Bild speichern",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpeg")]
                )
                if file_path:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    parent_dir = os.path.dirname(file_path)
                    save_dir = os.path.join(parent_dir,
                                            base_name
                                            )
                    os.makedirs(save_dir,
                                exist_ok=True
                                )
                    image_path = os.path.join(save_dir,
                                              f"{base_name}.png"
                                              )
                    self.current_image.save(image_path)
                    json_path = os.path.join(save_dir,
                                             f"{base_name}.json"
                                             )
                    save_metadata_to_json(json_path,
                                          seed_tensor=self.current_seed
                                          )
                    messagebox.showinfo("Erfolg", "Bild und Seed als JSON erfolgreich gespeichert!")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern: {str(e)}")
        return None

    """
    Funktion hinter dem Help-Button.
    """

    def show_help(self) -> None:
        help_text = """GAN-CARD Hilfe:

Generieren Tab:
- Generiert zufällige Bilder mit verstecktem Seed
- Grußformel optional hinzufügen
- Position der Grußformel wählen (↖↗↙↘)

Reproduzieren Tab:
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

        messagebox.showinfo("Hilfe", help_text)
        return None

    """
    Funktion hinter dem Credits-Button.
    """

    def show_credits(self) -> None:
        messagebox.showinfo(
            'Credits',
            '             GAN-CARD v8.12\n Dennis Becker & Lukas Laaser\nEngineering ML-based Systems\n\nMit Steganografie & Grußformel-Feature\nGoogle Font: Old Standard TT'
        )
        return None


def main() -> None:
    app = GANCardGUI()
    return None


if __name__ == '__main__':
    main()