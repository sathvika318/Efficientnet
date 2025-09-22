# Brain Tumor Detection Using EfficientNetB0

This project implements a deep learning model for **brain tumor classification** using MRI images. The model leverages **EfficientNetB0** with transfer learning and custom classification layers to identify four categories:

- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

---

## Dataset

- The dataset is already split into **Training**, **Validation**, and **Testing** folders.  
- Each split contains subfolders for the four classes.  
- Images are preprocessed and resized to **224x224** pixels.  

---

## Features

- **Data Augmentation**: Random flips, rotations, and zooms applied to training set.  
- **Transfer Learning**: EfficientNetB0 pretrained on ImageNet as the base model.  
- **Custom Classifier Head**: Dense + Dropout layers added on top of EfficientNet.  
- **Two-Stage Training**:
  1. **Feature Extraction**: Freeze base model, train only classifier head.  
  2. **Fine-Tuning**: Unfreeze base model and fine-tune with smaller learning rate.  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC & Precision-Recall curves.  

---

## Requirements

- Python 3.x  
- TensorFlow 2.x  
- OpenCV (`cv2`)  
- Matplotlib  
- Seaborn  
- scikit-learn  
- NumPy  
- Pandas  

Install dependencies with:

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn numpy pandas
