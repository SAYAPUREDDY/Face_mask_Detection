# Face Mask Detection Using VGG19

This repository contains code for a face mask detection model built using the VGG19 architecture and OpenCV for face detection. The model is trained to classify images as either "with mask" or "without mask" and includes functionality for face detection and mask classification in real-time.

## Project Overview

The main goal of this project is to build a deep learning model capable of detecting whether a person is wearing a face mask or not. The model uses the VGG19 architecture, which is pre-trained on the ImageNet dataset, and fine-tuned for binary classification (mask vs. no mask). OpenCV is utilized for face detection, which enables the identification and classification of faces in images.

## Dataset

The dataset used for training and testing the model consists of two classes:

1. **With Mask**: Images of people wearing face masks.
2. **Without Mask**: Images of people without face masks.

### Data Paths
- **With Mask**: `../input/face-mask-12k-images-dataset/Face Mask Dataset/Train/WithMask`
- **Without Mask**: `../input/human-faces/Humans`

## Code Overview

### Dependencies

- **Pandas**: For handling dataframes.
- **NumPy**: For numerical operations.
- **Seaborn & Matplotlib**: For data visualization.
- **Scikit-learn**: For shuffling the dataset.
- **Keras**: For building and training the deep learning model.
- **OpenCV**: For face detection.

### Main Components

1. **Data Loading and Preprocessing**:
   - Images are loaded from the specified directories and their paths are stored in dataframes.
   - Data is shuffled to ensure randomness.

2. **Data Augmentation**:
   - Used to improve model generalization by applying transformations such as rotation, shifting, and flipping.

3. **VGG19 Model**:
   - The VGG19 model is used as a feature extractor with the top layers removed. A custom classification head is added for binary classification.

4. **Training and Evaluation**:
   - The model is trained using the augmented data, and performance is evaluated on validation and test datasets.

5. **Face Detection**:
   - OpenCV's Haar Cascade Classifier is used to detect faces in an image.
   - Detected faces are passed through the trained model to predict whether the person is wearing a mask.

6. **Prediction**:
   - The model predicts the class (mask or no mask) for detected faces and annotates the image with the results.

### Training and Validation Results

- **Training Accuracy**: ~96%
- **Validation Accuracy**: ~96%
- **Test Accuracy**: ~96%

### Example Usage

Here are some example steps to run the code:

1. **Data Visualization**:
   - Visualize the distribution of the classes using `sns.countplot`.

2. **Plotting Images**:
   - Display sample images from the dataset with and without masks.

3. **Model Training**:
   - Train the model using the training set with real-time data augmentation.

4. **Face Detection**:
   - Detect faces in an image and classify them as "with mask" or "without mask".

5. **Real-Time Detection**:
   - Use the trained model for real-time mask detection in images.

### Code Execution

1. **Training the Model**:
   - To train the model, execute the provided code in a Jupyter notebook or any Python environment.

2. **Face Detection and Mask Classification**:
   - After training, the model can be used to detect and classify faces in images. Simply load an image, detect faces using OpenCV, and pass them through the model for prediction.

### Conclusion
This project demonstrates the use of deep learning and computer vision techniques for face mask detection. The model achieves high accuracy and can be further improved with more data and fine-tuning. The combination of the VGG19 architecture and data augmentation has proven to be effective in distinguishing between masked and unmasked faces.

### Acknowledgements
- Keras: For the deep learning framework.
- OpenCV: For the face detection functionality.
- VGG19: For the pre-trained model architecture.






