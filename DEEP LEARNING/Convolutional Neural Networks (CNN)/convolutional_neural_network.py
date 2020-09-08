# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255, # corresonds to feature scaling part (converts pixel values b/w 0 and 1)
                                   shear_range = 0.2, # corresonds to shearing (geometrical transformation) 
                                   zoom_range = 0.2, # some sort of random zoom that we apply
                                   horizontal_flip = True) # horizontally flips the images
# imaga data generator prevents overfitting
# ( becoz here we don't want a correlation b/w IDV and DV but a pattern in pixels which req lot of images, hence chance of overfitting)
# data augmentation- creating batches of images and applying transformations
#                  - it is a technique to enrich our dataset

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), # the size of images that is expected in our CNN model
                                                 batch_size = 32,# size of batches with random samples of our images
                                                 class_mode = 'binary') # indicating that DV is binary

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential() # for sequence of layers in CNN
# now comes the exciting step of adding different layers

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
# this step comprises of applying several feature detectors to the input image which gives several feature maps 
# 'filters' is the no. of feature detectors that we will apply, also corresponds to no. of feature maps we want to create
# 'kernel_size' is the dimension of feature detector (i.e. 3x3)
# 'padding' is just to specify how feature detectors will handle the border of the input image
# 'input_shape' shape of input image (we have gto force them to have same format i.e. fixed size)
# here 3 is the no. of channels, it will be 1 for B/W, and 64x64 is the resolution enough for classifying images on CPU
# we are using 'activation function' to be sure there is no -ve pixel value in feature maps (i.e. to have non-linearity)

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
# just reducing the size of feature maps
# 'pool_size=2' size of the feature maps is divided by 2(i.e. reduce complexity without reducing the performance)

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
# we can inc. no. of features and double it each time.(i.e. by inc kernel-size)
# to inc. accuracy, we can either add a convolutional layer or a full connection layer

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())
# converting the pooled feature maps into a single vector
# the high numbers in feature maps detect the spatial structure of our images, hence we don't loose the structure during flattening

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) # to add fully connected layers(hidden layers)
# 'units' is the no. of nodes in hidden layer

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# since binary O/P (dog or cat) hence sigmoid, otherwise softmax
# 'units=1' as it will contain predicted probability of one class

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# 'optimizer = 'adam' for stochastic gradient
# 'loss = 'binary_crossentropy' as it correspnds to logarithmic loss, also we have binary outcome

# Training the CNN on the Training set and evaluating it on the Test set
# Image Augmentation/Preprocessing (fitting our CNN to all our images)
cnn.fit_generator(training_set,
                  steps_per_epoch = 334,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 334)