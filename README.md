# DeepVision Traffic Sign Recognition

A deep learning-based traffic sign recognition system that uses a **Convolutional Neural Network (CNN)** to classify traffic sign images into **43 categories**.

## Objective

To develop an image classification model that can automatically recognize traffic signs from uploaded images.

## Model

- **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)
- **Model:** Convolutional Neural Network (CNN)
- **Input Size:** 64 × 64 × 3
- **Number of Classes:** 43
- **Framework:** TensorFlow / Keras

## Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **95.14%** |
| Macro F1 Score | **92.66%** |
| Weighted F1 Score | **95.04%** |

## Explainability

The project includes **Grad-CAM** to visualize the important regions of an image that influenced the model's prediction.

## Deployment

The trained model is deployed using **Streamlit**, allowing users to upload a traffic sign image and receive a prediction with confidence.

**Live Demo:**  
https://deepvision-traffic-sign-recognition.streamlit.app/

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn
- Streamlit
- Git & GitHub

## Author

**M Ankitha**

Data Science & Machine Learning Enthusiast
