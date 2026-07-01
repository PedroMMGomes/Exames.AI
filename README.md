<p align="center">
  <img src="readme-hero.png" alt="Exames.AI — classificador de imagens médicas" width="100%">
</p>

<h1 align="center">🩻 Exames.AI — Classificador de Imagens Médicas</h1>
<p align="center"><strong>Projeto experimental de visão computacional: classifica imagens em <em>Benigno / Câncer / Normal</em> com modelo Keras (Transfer Learning) e GUI Tkinter.</strong></p>

<p align="center">
  <img alt="lang" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat-square">
  <img alt="dl" src="https://img.shields.io/badge/Keras-TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=flat-square">
  <img alt="cv" src="https://img.shields.io/badge/visão-PIL%20%2F%20NumPy-0E7490?style=flat-square">
  <img alt="gui" src="https://img.shields.io/badge/GUI-Tkinter-2D7FF9?style=flat-square">
</p>

---

Projeto inicial de estudo em **Machine Learning aplicado à saúde**. Um modelo de classificação de imagens (treinado por Transfer Learning) distingue três classes de exames:

| ID | Classe |
|----|--------|
| 0 | Benign |
| 1 | Cancer |
| 2 | Normal |

## 🚀 Como usar

```bash
pip install tensorflow keras pillow numpy
python GPT1.py        # abre a GUI Tkinter — selecione imagens para classificar
```

O modelo treinado está em `keras_model.h5` e os rótulos em `labels.txt`.

## 📁 Estrutura

- `GPT1.py` … `GPT4.py` — iterações do classificador (GUI Tkinter + Keras)
- `tensor1.py` — experimentos com TensorFlow
- `keras_model.h5` / `labels.txt` — modelo e rótulos (Treinável via Teachable Machine)
