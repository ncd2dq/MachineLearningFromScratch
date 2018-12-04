# Feed Forward Neural Network



# Forward Pass

Let's take a simple example data: [a, b] and feed it through a 3 layer network of size (2, 3) (3, 3) (3, 1).

For matrix multiplication we take the rows of the first matrix and dot product them against the columns of the second marix.

Note for matrix - Name(rows, cols) - A(1, 3) to be multiplied by B(3, 1) the cols of A must match the rows of B.

#### First Layer ####
(1, 2) dot (2, 3) yields (1, 3) <br>
```
a b * i j k  -->  ai+bl aj+bm ak+an
      l m n  -->
```


#### Second Layer ####
(1, 3) dot (3, 3) yields (1, 3) <br>
```
a b c * i j k --> ai+bl+co aj+bm+cp ck+cn+cq
	l m n -->
	o p q -->
```

#### Third Layer ####
(1, 3) dot (3, 1) yields (1, 1) <br>
```
a b c * i --> ai+bj+ck
	j -->
	k -->
```

In a way, you can think of the Neural Network as transforming the feature data into label data. Notice that in the above example we started out with 1 row of feature data with 2 features - (1, 2) - and we ended up with 1 row of label data with 1 label - s(1, 1)

# Back Propogation

Back propogation is the process by which the network updates the weights of all the synapses within itself by propogating the error of the final output backwards through the network. This involves a bit of calculus, but is rather simple to implement in code for a feed forward network. In short, we multiply the error by the derivative of the activation function at a given layer and use this to calculate how much we want to update our weights.

By multiplying the error by the derivative of the activation function, we are asking the simple question - for a given node, how much is our activation function changing at the values in our current layer. Let's look at the sigmoid function, for example: 

When the value of x is extremely positive (NN is very certain in the positive direction) and when the y value is extremely negative (NN is very certain in the negative direction), the derivative of the sigmoid is very small. This means that that weights in that node will not be updated very much. This makes sense because we don't want to change neurons if they are very certain, only those with less certain outputs (which correclates to areas where the derivative is very large).