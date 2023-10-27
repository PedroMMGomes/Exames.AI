import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import Image, ImageOps, ImageTk
import numpy as np

def predict_images():
    global img_label
    global result_label
    global confidence_label

    filepaths = filedialog.askopenfilenames(title="Selecione imagens", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")])
    
    for filepath in filepaths:
        # Carregar imagem
        image = Image.open(filepath).convert("RGB")
        
        # Atualizar imagem no GUI
        img = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo)
        img_label.image = photo

        # Redimensionar e cortar
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Atualizar rótulo de resultado
        result_label.config(text=f"Resultado: {class_name}")
        
        # Atualizar rótulo de confiança e cor
        confidence_label.config(text=f"Precisão: {confidence_score:.2f}%")
        if 90 <= confidence_score <= 100:
            confidence_label.config(fg="green")
        elif 60 <= confidence_score < 90:
            confidence_label.config(fg="yellow")
        else:
            confidence_label.config(fg="red")
        
        results.insert(tk.END, f"{filepath}: Class - {class_name}, Confidence - {confidence_score:.2f}")

np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

root = tk.Tk()
root.title("mammogram.AI - AtomDev")

btn_select = tk.Button(root, text="Selecione imagens", command=predict_images)
btn_select.pack(pady=20, padx=20, side=tk.LEFT)

# Mostrar imagem
img_label = tk.Label(root)
img_label.pack(pady=20, padx=20, side=tk.LEFT)

# Rótulo para o resultado
result_label = tk.Label(root, text="Resultado: ", font=("Arial", 16))
result_label.pack(pady=10, padx=20, side=tk.LEFT)

# Rótulo para a precisão
confidence_label = tk.Label(root, text="Precisão: ", font=("Arial", 16))
confidence_label.pack(pady=10, padx=20, side=tk.LEFT)

results = tk.Listbox(root, width=120, height=20)
results.pack(pady=20, padx=20, side=tk.RIGHT)

root.mainloop()