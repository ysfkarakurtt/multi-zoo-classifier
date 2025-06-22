# MultiZoo Animal Classifier

##  Description

MultiZoo is a desktop-based animal image classification application developed using deep learning techniques. The system utilizes a Vision Transformer (ViT-Tiny) architecture to classify animal species based on images provided by the user.

##  Features

* Upload any animal image from your computer.
* Classify the animal species using a pre-trained ViT model.
* Display prediction label and model confidence score.
* Lightweight and easy-to-use GUI with Tkinter.


---

##  Technologies Used

* **Model Architecture:** Vision Transformer (ViT-Tiny)
* **Libraries & Frameworks:**

  * PyTorch
  * `timm` (for Vision Transformer models)
  * Tkinter (GUI)
* **Development Environment:** Google Colab, Python 3.10+
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

---

##  Model Training

* **Dataset:** MultiZoo Dataset provided by course coordinators
* **Split:** 80% Training, 20% Validation, separate Test Set
* **Augmentation:** Resize, normalization, random flip
* **Optimizer:** Adam
* **Loss Function:** Cross Entropy Loss
* **Performance Example:**
---
##  Model Evaluation Metrics

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 0.946296  |
| Precision  | 0.958203  |
| Recall     | 0.946296  |
| F1-Score   | 0.945695  |

<p align="center">
  <img src="https://github.com/user-attachments/assets/2c0993b4-eff0-4218-87ff-5ee33b170412" width="400" alt="Screenshot 4" />
  <img src="https://github.com/user-attachments/assets/aa041af8-08bb-4c8e-8c81-93368dd427ad" width="400" alt="Screenshot 5" />
</p>

---

##  How to Use

1. Run the Python application.
2. Upload an image via the GUI.
3. The model processes the image and predicts the animal class.
4. The prediction and confidence score are displayed.

---

##  Requirements

* Python >= 3.10
* PyTorch
* torchvision
* timm
* tkinter

---

##  Notes

* Predictions are based on image classification using a pre-trained ViT model.
* This project was evaluated based on GUI integration, accuracy, and usability.

---


##  Sample Screenshots
<p align="center">
  <img src="https://github.com/user-attachments/assets/40342c7a-b46d-4e45-be66-f4b33a853302" width="400" alt="Screenshot 1" />
  <img src="https://github.com/user-attachments/assets/58fbea59-1a52-4667-b97e-d2689ffe257f" width="400" alt="Screenshot 2" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/a675b640-79d6-45e1-8144-41af434c184a" width="400" alt="Screenshot 3" />
</p>
---
