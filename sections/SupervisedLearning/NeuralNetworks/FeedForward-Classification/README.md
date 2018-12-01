# Feed Forward Neural Network



# Forward Pass

Let's take a simple example data: [a, b] and feed it through a 3 layer network of size (2, 3) (3, 3) (3, 1).

For matrix multiplication we take the rows of the first matrix and dot product them against the columns of the second marix.

Note for matrix - Name(rows, cols) - A(1, 3) to be multiplied by B(3, 1) the cols of A must match the rows of B.

#### First Layer ####
(1, 2) dot (2, 3) yields (1, 3) <br>
`
a b * i j k  -->  ai+bl aj+bm ak+an
      l m n  -->
`


#### Second Layer ####
(1, 3) dot (3, 3) yields (1, 3) <br>
`
a b c * i j k --> ai+bl+co aj+bm+cp ck+cn+cq
		l m n -->
		o p q -->
`

#### Third Layer ####
(1, 3) dot (3, 1) yields (1, 1) <br>
`
a b c * i --> ai+bj+ck
		j -->
		k -->
`

# Back Propogation

Back propogation is the process by which the network updates the weights of all the synapses within itself by propogating the error of the final output backwards through the network. This involves a bit of calculus, but is rather simple to implement in code for a feed forward network.
