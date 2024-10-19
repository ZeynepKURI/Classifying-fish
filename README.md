
# Deep Learning Fish Classification Project

## Project Overview
This project focuses on classifying fish species using the **A Large Scale Fish Dataset**. The dataset contains thousands of labeled images representing various fish species. The aim is to accurately classify fish images into their corresponding species, demonstrating the effectiveness of convolutional neural networks (CNNs) in image recognition tasks.

## Dataset
The **A Large Scale Fish Dataset** includes various fish species collected from different environments. The dataset has been divided into training, validation, and test sets for comprehensive evaluation of model performance. Key features of the dataset include:
- **Number of Classes**: 9 different fish species
- **Image Size**: Images have been resized to a consistent dimension of 150x150 pixels to maintain uniformity.
- **Data Augmentation**: Various transformations such as rotation, width/height shifting, and horizontal flipping have been applied to enhance the model’s generalization capability.

## Methodology
1. **Data Preparation**:
   - The dataset was loaded into a DataFrame containing image paths and corresponding labels.
   - Data augmentation techniques were used to increase the model's robustness.
   - 
### Data Splitting
The dataset has been divided into three main parts: training, validation, and testing sets:

1. **Training Set**: This dataset is used for the model's learning process. It helps the model learn to recognize different fish species and their characteristics.
2. **Validation Set**: This set is used to evaluate the model during the training process, allowing for assessment of the model's performance and prevention of overfitting.
3. **Test Set**: This set is utilized for the final evaluation of the model's performance. It consists of data that has not been used during training or validation.

The data splitting was accomplished using the `train_test_split` function, ensuring a balanced selection of data points in each set.

## Data Augmentation

To enhance the model's generalization capability and improve performance, various data augmentation techniques were applied to the training dataset. These techniques are as follows:

1. **Rescale**: 
   - `rescale=1/255`: Scales the pixel values of the images from a range of 0-255 to a range of 0-1, normalizing the input data.

2. **Rotation Range**: 
   - `rotation_range=40`: Randomly rotates the images by an angle between 0 and 40 degrees, introducing variability in the training data.

3. **Width Shift Range**: 
   - `width_shift_range=0.2`: Randomly shifts the images along the width by up to 20%, allowing the model to learn from different horizontal positions of the objects.

4. **Height Shift Range**: 
   - `height_shift_range=0.2`: Randomly shifts the images along the height by up to 20%, enabling the model to handle vertical displacements.

5. **Shear Range**: 
   - `shear_range=0.2`: Applies random shear transformations to the images, altering their shape and perspective.

6. **Zoom Range**: 
   - `zoom_range=0.2`: Randomly zooms in or out on the images by up to 20%, creating variations in the scale of the fish.

7. **Horizontal Flip**: 
   - `horizontal_flip=True`: Randomly flips the images horizontally, enhancing the diversity of the training set by reflecting the images.

These augmentation techniques were essential in preventing overfitting and improving the model's ability to generalize to new, unseen data.

2. **Model Architecture**:
   - A CNN architecture was designed to extract relevant features from images.
   - The model architecture consists of the following components:
     - Convolutional layers for feature extraction
     - Max pooling layers for dimensionality reduction
     - A dense layer with softmax activation to provide probabilities for each class.

3. **Model Compilation and Training**:
   - The model was compiled using the Adam optimization algorithm and categorical cross-entropy loss function.
   - The model was trained on the training set while evaluating its performance on the validation set.
  
   - 
# Model Evaluation Metrics

## 1. Accuracy
Accuracy measures the ratio of correctly classified instances to the total instances. A high accuracy indicates that the model performs well overall.

## 2. Precision
Precision measures the ratio of true positive predictions to the total predicted positives. High precision indicates that the model has a low rate of false positive predictions.

## 3. Recall
Recall measures the ratio of true positive predictions to the actual positives. High recall indicates that the model is good at identifying positive instances.

## 4. F1-Score
F1-Score is a metric that provides a balance between precision and recall. It is useful when both precision and recall are important.

## 5. Confusion Matrix
The confusion matrix visualizes the model's correct and incorrect predictions for each class. It helps to understand the strengths and weaknesses of the model.

## Results
- An accuracy of %X was achieved on the validation set, demonstrating the model's effectiveness in classifying fish species.
- Additional metrics (precision, recall, and F1 score) were used for a comprehensive evaluation of the model's performance.

## Conclusion
This project highlights the application of deep learning techniques to image classification tasks, particularly in the field of marine biology. Successfully classifying fish species can aid in biodiversity studies, ecological research, and the development of conservation strategies.


## Requirements
- TensorFlow
- Keras
- Matplotlib


## How to Run
1. Clone the repository.
2. Install the required packages.
3. Run the project using Jupyter Notebook or Python scripts.

## Acknowledgments
- A Large Scale Fish Dataset:  https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data 
- Special thanks to all contributors and the community for their continuous support.

## Kaggle Projesi
Bu projeye ait kod ve veriler için Kaggle sayfasını ziyaret edebilirsiniz: [Kaggle Projesi Linki](https://www.kaggle.com/code/zeynepkuri/notebook1f6)

