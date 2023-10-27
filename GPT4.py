import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import Image, ImageOps, ImageTk
import numpy as np
from tkinter import ttk
# Função para carregar e fazer previsões
def predict_images():
    global img_label
    global result_label
    global confidence_label

    filepaths = filedialog.askopenfilenames(title="Selecione imagens", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")])
    
    for filepath in filepaths:
        # Carregar imagem
        image = Image.open(filepath).convert("RGB")
        
        # Atualizar imagem no GUI
        img = ImageOps.fit(image, (250, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo)
        img_label.image = photo

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index] * 100

        # Cores para os resultados
        colors_for_classes = {
            "Normal": "green",
            "Benign": "yellow",
            "Cancer": "red"
            
        }
        
        # Atualizar rótulo de resultado e cor
        result_label.config(text=f"Resultado: {class_name}", fg=colors_for_classes.get(class_name, "black"))
        
        # Atualizar rótulo de confiança e cor
        confidence_label.config(text=f"Precisão: {confidence_score:.2f}%")
        if 90 <= confidence_score <= 100:
            confidence_color = "green"
        elif 60 <= confidence_score < 90:
            confidence_color = "yellow"
        else:
            confidence_color = "red"
        confidence_label.config(fg=confidence_color)
        
        results.insert(tk.END, f"{filepath}: Class - {class_name}, Confidence - {confidence_score:.2f}")

# Configurações iniciais
np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# GUI
root = tk.Tk()
root.title("mammogram.AI - AtomDev")

# Aumente o tamanho da imagem
img_size = (300, 300)

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=20)

# Use o ttk.Button para um botão mais estilizado
style = ttk.Style()
style.configure("TButton", font=("Trebuchet MS", 16), padding=10)
btn_select = ttk.Button(left_frame, text="Selecione imagens", command=predict_images)
btn_select.pack(pady=20, padx=10)

img_label = tk.Label(left_frame)
img_label.pack(pady=20, padx=10)

result_label = tk.Label(left_frame, text="Resultado: ", font=("Trebuchet MS", 20))
result_label.pack(pady=20, padx=10)

confidence_label = tk.Label(left_frame, text="Precisão: ", font=("Trebuchet MS", 20))
confidence_label.pack(pady=20, padx=10)

results = tk.Listbox(root, width=100, height=30)
results.pack(pady=20, padx=10, side=tk.RIGHT)

root.mainloop()