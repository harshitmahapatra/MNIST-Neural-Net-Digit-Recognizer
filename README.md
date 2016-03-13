# MNIST-Neural-Net-Digit-Recognizer
The script is for classification of a subset of MNIST handwritten digits database obtained from the url: http://pjreddie.com/projects/mnist-in-csv/ <br/>
The procedure followed is as follows:<br/>
1.The training set was divided into training subset and Cross Validation set by random sampling.<br/>
2.Each set was then feature normalized.<br/>
3.A single hidden layer neural network using package "nnet" was trained on the training subset for various parameter values and corresponding CV set errors were stored.<br/>
4.The parameter value with least amount of CV set error was selected and was used to make predictions on the test set.<br/>
