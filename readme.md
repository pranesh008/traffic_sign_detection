# Traffic Sign Detection using CNN

## Overview
This project is a **Traffic Sign Detection System** that uses **Convolutional Neural Networks (CNNs)** to identify and classify traffic signs from images. The model is trained using the **Keras** and **TensorFlow** frameworks in Python.

## Features
- Detects and classifies multiple traffic signs.
- Trained on a robust dataset using CNN architecture.
- Achieves high accuracy on validation and test data.
- Provides visualization of predictions.

## Dataset
The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which consists of 43 classes of traffic signs. The dataset includes:
- Training images with corresponding labels.
- Validation and test datasets for evaluation.

## Technologies Used
- **Python**
- **TensorFlow & Keras**
- **OpenCV**
- **NumPy & Pandas**
- **Matplotlib & Seaborn**

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/traffic-sign-detection.git
   cd traffic-sign-detection
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Model Architecture
- **Input Layer**: Image preprocessing & resizing.
- **Convolutional Layers**: Extract features using filters.
- **Pooling Layers**: Reduce dimensionality.
- **Fully Connected Layers**: Classify signs into categories.
- **Output Layer**: Softmax activation for multi-class classification.

## Training the Model
Run the following command to train the model:
```sh
python train.py
```

## Testing and Evaluation
To test the model on new images:
```sh
python predict.py --image path_to_image.jpg
```

## Results
- Achieved an accuracy of **87%** on the test dataset.
- Model generalizes well to unseen traffic sign images.

## Future Improvements
- Improve accuracy using transfer learning (e.g., ResNet, MobileNet).
- Extend dataset for better generalization.
- Deploy the model as a web application.

## Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, reach out via email at `praneshspuri@gmail.com` or create an issue in the repository.


