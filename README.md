# Sports-Image-Classification

The first model: (Simple CNN)
Pre-processing:
In this model First we read all data set and resize each image
to (50, 50 ,3) in function create_data() then we create for each image their label using function create_label(image_name)
after we get our we split it to train data and test data
• 80% Train data
• 20% Test data
Then we get X and Y
For input data and there target:
---------------------------------------------------------------------------------------
The main model here is simple CNN. CNN is a type of deep learning model for processing data that has a grid pattern, such as images
• CNN is a class of deep, feed-forward artificial neural networks, most commonly applied to analyzing visual imagery.
• CNNs, like neural networks, are made up of neurons with learnable weights and biases. Each neuron receives several inputs, takes a weighted sum over them, pass it through an activation function and responds with an output
Architecture:
CNN typically consist of convolutional layers, pooling layers, fully connected layer
Here we have 5 convolution layers and after each convolution layers we have pooling layer and in the end we connected them in fully connected layer
• 1st convolution layer in size (32, 32, 3)
• 2nd convolution layer in size (64, 64, 3)
• 3rd convolution layer in size (128, 128, 3)
• 4th convolution layer in size (32, 32, 3)
• 5th convolution layer in size (64, 64, 3)
With pooling layer after each convolution layer we used her activation function RELU between layers
The FC is the fully connected layer of neurons at the end of CNN. Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks and work in a similar way
FIT The model:
Here we check if we have old trained data and weights to start with if not we start to train our model with train data and test data
Optimizer: Adam
Number of epochs : 45
Learning rate : 0.001
After that we save new updated weight
Accuracy :
96% train accuracy
80% test accuracy
---------------------------------------------------------------------------------------
The second model: (ResNet50)
Pre-processing:
1-splitting data into 6 classes
2-Convert dataset into batches
3-Resize each image to (225,225,3)
---------------------------------------------------------------------------------------
it’s a deep-learning neural network that is used as a backbone for many computer vision like object detection, and image segmentation, residual networks aren’t a new type of CNN models it’s just an update of the simple CNN to get more efficiency and speed
Architecture: 1-Identity block: The identity block is the standard block used in ResNet and corresponds to the case where the input activation has the same dimension as the output activation.
2-Convolution block: We can use this type of block when the input and output dimensions don’t match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path.
The full implementation of the model:
Training summary:



Model training: ● Using Adam optimizer with a learning rate of 0.001 and categorical cross-entropy loss function with 70 epochs ● training accuracy= 99% and validation accuracy = 92% ● Testing accuracy=85
