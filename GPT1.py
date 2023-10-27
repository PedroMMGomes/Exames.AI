import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Função para carregar e fazer previsões
def predict_images():
    filepaths = filedialog.askopenfilenames(title="Selecione imagens", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")])
    
    for filepath in filepaths:
        # Carregar imagem
        image = Image.open(filepath).convert("RGB")
        
        # Redimensionar e cortar
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Converter imagem em numpy array
        image_array = np.asarray(image)
        
        # Normalizar
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Carregar imagem no array
        data[0] = normalized_image_array

        # Fazer previsão
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Adicionado .strip() para remover espaços e quebras de linha extras
        confidence_score = prediction[0][index]
        
        results.insert(tk.END, f"{filepath}: Class - {class_name}, Confidence - {confidence_score:.2f}")

# Configurações iniciais
np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Criar interface gráfica
root = tk.Tk()
root.title("mammogram.AI - AtomDev")

# Botão para selecionar imagens
btn_select = tk.Button(root, text="Selecione imagens", command=predict_images)
btn_select.pack(pady=20)

# Lista para mostrar resultados
results = tk.Listbox(root, width=200, height=20)
results.pack(pady=20)

root.mainloop()
