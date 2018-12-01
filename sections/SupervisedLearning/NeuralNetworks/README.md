# Neural Networks

A neural network can be considered a "universal function approximator". If you're an absolute math god you can think of these as similar to a [Taylor Series](https://en.wikipedia.org/wiki/Taylor_series) - essentially a way to take a function that already exists
and represent it in a different way. The main difference is that Neural Networks can approximate functions that we don't already have
standardized mathematical equations for!

The essence of Neural networks is learning how to associate a set of features (input) with a desired label (output). They create this association via training algorithms. The type of training we will be learning here is gradient discent via back propogation.

# A Simple Diagram

-Feature <br>
-Feature <br>
-Feature <br>
-Feature <br>
| <br>
| <br>
v <br>
-A Neaural Network <br>
| <br>
| <br>
v <br>
-Label <br>

# Features and Feature Selection

A feature is just an attribute of your data that you think influences your desired result. Imagine if you were tasked with coming up with the value of a house. What data would you want to help you out! Those are your features. If you're building a neural network to appraise the value of a house you might use the following features:
1. \# of Rooms
2. Sq. Ft.
3. State
4. City
5. \# of bathrooms
6. Garage Space
7. Has a pool or not

The magic of Neural Networks is that they will learn which features and feature combinations are the most important. Although we created a list above of features we think are important, maybe the only thing that matters is '# of Rooms' and 'Sq. Ft.'. The Neural Network is a great tool to figure out some combination of features that are important that a human may not easily figure out.

The above example was rather simple, sometimes it's really difficult to determine what features to give your Neural Network and there is a whole field of study on Feature Selection that I wont dive into, but just know that can be an entire tutorial in itself.

# Training

We can break training down into two parts:
* Forward Pass
* Back Propogation

I will generally describe it here and go more in-depth in the sections on specific types of neural networks as the architecture will affect the algorithm.

# Forward Pass

A forward pass consists of feeding your feature data through all the layers of your neural network. Mathemtaically this amounts to a lot of matrix multiplication between your feature data matrix and the weight matrices of the layers in your network. Conceptually, this means the weights in your neural network will activate to a varying amount and pass a signal through the network depending on how closely they recognize a pattern.

Depending on the type of network, this "forward" pass can get quite intricate.


# Back Propogation

Back propogation is the process by which the network updates the weights of all the synapses within itself by propogating the error of the final output backwards through the network. This involves a bit of calculus, but a full understanding of the math is not necessary to grasp the concept, nor implement the algorithm. Mathematically, you are taking your error and multiplying it by the derivative of your activation function at every given layer and using that outcome to update your weights. Conceptually, you are assigning a "weighted fault" if you will to each node in the network. Nodes that played a larger part in the error will change more than those that didn't.