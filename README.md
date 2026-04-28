# 🎨 AI Image Segmentation Studio (U-Net vs Baseline)

An interactive deep learning project that demonstrates how **model architecture impacts segmentation quality**, both visually and quantitatively.

---

## 🌐 Live Demo

👉 https://creative-image-segmentation-6gczdhydvheogigrzftxr3.streamlit.app

---

## 🧠 What this project does

This project explores **image segmentation** using two different approaches:

* 🧩 **U-Net** — a state-of-the-art architecture for pixel-level prediction  
* 🧪 **Baseline CNN** — a simpler model for comparison  

It allows you to:

- Upload an image  
- Run segmentation in real-time  
- Compare outputs visually  
- Evaluate performance using metrics  

---

## 🎯 Why this matters

Model architecture is not just an implementation detail —  
it fundamentally affects **how well a model understands visual structure**.

This project shows:

- how **U-Net captures spatial context**
- why **naive CNNs struggle with segmentation**
- how performance differences appear **both numerically and visually**

👉 A practical demonstration of **deep learning design decisions**

---

## 📊 Results

| Model     | IoU  | Dice |
|----------|------|------|
| U-Net    | 0.59 | 0.74 |
| Baseline | 0.30 | 0.46 |

---

## 📸 Preview

![Demo](results/comparisons/compare_20260425_213222.png)

---

## 🚀 Features

* 🧠 U-Net vs Baseline comparison
* 🎯 Binary image segmentation
* 📊 IoU & Dice evaluation
* 🖼️ Visual side-by-side results
* ⚡ Real-time inference (Streamlit)
* 🌐 Interactive web app

---

## 🛠 Tech Stack

* Python
* PyTorch
* Streamlit
* NumPy
* OpenCV

---

## ⚙️ How it works

1. Image is preprocessed and resized  
2. Model predicts segmentation mask  
3. Output is post-processed  
4. Metrics (IoU, Dice) are computed  
5. Results are displayed interactively  

---

## ▶️ Run locally

```bash
git clone https://github.com/dealmeidaferreiraAlexandra/creative-image-segmentation.git
cd creative-image-segmentation

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

⚠️ Notes
Model trained on Oxford-IIIT Pet Dataset
Performance may vary on real-world images
Baseline model included for educational comparison


👩‍💻 Author

Developed by Alexandra de Almeida Ferreira
GitHub: https://github.com/dealmeidaferreiraAlexandra
LinkedIn: https://www.linkedin.com/in/dealmeidaferreira

📄 License

This project is licensed under the MIT License.
