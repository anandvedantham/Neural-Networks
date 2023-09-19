# Neural-Networks

Neural networks, also known as artificial neural networks (ANNs) or simply "neural nets," are a class of machine learning models inspired by the structure and functioning of the human brain. They are a fundamental component of deep learning, a subfield of machine learning.The basic building blocks of neural networks are artificial neurons, also called nodes or units. These neurons are analogous to the neurons in the human brain. Each neuron receives input signals, performs a computation on these inputs, and produces an output signal.

Neurons in a neural network are organized into layers. There are typically three types of layers in a neural network. Input layer receives the initial input data and passes it to the subsequent layers. Hidden layers, which can be one or more, are responsible for performing computations on the input data. They extract features and patterns from the input data.  The output layer produces the network's output, which is typically a prediction, classification, or some other relevant result.

## Implementation
- Importing the required Libraries and reading the dataset
- Performing EDA on the dataset
- Visualizations
- Data Preprocessing
- Splitting the data into Train and Test data
- Standardization of the data
- Tuning of the Hyperparameters
  - Batch Size
  - Epochs
  - Learning Rate
  - Number of Layers and Neurons
- Building the Model using Neural Networks
  - Training the model with best parameters
  - Model Evaluation Train and Test Error
- Predictions and Finding the accuracy

## Packages Used
- pandas
- numpy
- matplotlib.pyplot
- seaborn
- warnings
- from sklearn.preprocessing import StandardScaler
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import accuracy_score, confusion_matrix
- from sklearn.model_selection import GridSearchCV, KFold
- import keras, tensorflow
- from keras.models import Sequential
- from keras.layers import Dense, Dropout
- from keras_tuner.tuners import RandomSearch
- from scikeras.wrappers import KerasClassifier, KerasRegressor
- from tensorflow.keras.optimizers import Adam
