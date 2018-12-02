# Machine Learning From Scratch

This repo is intended to give you a good understanding of the basics of how to implement Machine Learning
from scratch (no dependencies except numpy). 

There are many amazing libraries out there like TensorFlow, PyTorch, and scikit-learn - to name a few - that use GPU optimization, implement the latest algorithms, and are written in C to deliver the best performance. 
The only problem with these libraries is that they abstract the HELL out of everything. 

Abstraction is great if you're a professional who needs to deliver production code that is reliable, 
fast, and scalable, but if you're just starting out with Machine Learning it can make the "hands on" approach a lot less... well... "hands on".

I found that during my journey through Machine Learning I kept asking myself, "but how does this work??",
and although I read through many repo's and ML libraries to try and understand, everything seemed
so obscure. I hope to collect just the raw essence of ML here.

## Getting Started

While we are not going to use dependencies outside of Numpy, I recommend making a virtual environment
just to practice.

### Prerequisites

1) A virtual environment
2) Python 3.x
3) Numpy
4) Watch this video series: https://www.youtube.com/watch?v=aircAruvnKk

For Windows
```
python -m venv venv
venv\Scripts\activate
pip install numpy matplotlib os random typing
```

For MacOS
```
python -m venv venv
. venv/bin/activate
pip install numpy matplotlib os random typing
```

## Contents

* [Supervised Learning](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/SupervisedLearning)
	* [Neural Networks](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/SupervisedLearning/NeuralNetworks)
		* [Different Types](https://cdn-images-1.medium.com/max/1000/1*cuTSPlTq0a_327iTPJyD-Q.png)
		* [Feed Forward  - Classification](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/SupervisedLearning/NeuralNetworks/FeedForward-Classification)
		* [Feed Forward - Regression](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/SupervisedLearning/NeuralNetworks/FeedForward-Regression)
		* [Recurrent](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/SupervisedLearning/NeuralNetworks/Recurrent)
		* [Convolutional](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/SupervisedLearning/NeuralNetworks/Convolutional)	
	* [Decision Trees](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/SupervisedLearning/DecisionTrees)
	* [Support Vector Machines](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/SupervisedLearning/SupportVectorMachines)


* [Unsupervised Learning](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/UnsupervisedLearning)
	* [K-Means Clustering](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/UnsupervisedLearning/KMeansClustering)


* [Reinforcement Learning](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/ReinforcementLearning)
	* [Q-Learning](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/ReinforcementLearning/QLearning)
	* [Deep Q-Learning (Deep Reinforcement Learning)](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/ReinforcementLearning/DeepQLearning)


* [Evolutionary Algorithms](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/EvolutionaryAlgorithms)
	* [Genetic Algorithm](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/EvolutionaryAlgorithms/GeneticAlgorithm)
	* [Neuro-Evolutionary Network (Neural Networks + Evolutionary Algorithms)](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/EvolutionaryAlgorithms/NeuroEvolutionaryNetwork)

* [Machine Learning Terminology](https://github.com/ncd2dq/MachineLearningFromScratch/tree/master/sections/MachineLearningTerminology)

## Recommended Initial Order

I recommend going through these 4 first to get a good base level understanding of machine learning. After this list, the journey is all yours!

1. Evolutionary Algorithms
2. Feed Forward - Classification
3. Neuro-Evolutionary Network
4. Q-Learning

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Nicholas Dima** - *Tutorials/Code* - [Me](www.nickdima.com)

## License

Do anything you'd like with this, as long as you learn something along the way!

## Acknowledgments

* The blog post that got me started with ML and inspired this tutorial series - [I am trask](https://iamtrask.github.io/2015/07/12/basic-python-network/)
* Thank you [The Coding Train's Daniel Schiffman](https://www.youtube.com/channel/UCvjgXvBlbQiydffZU7m1_aw) for being the reason I stuck with coding
* Thank you [Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) for your dedication to ML videos
* Thank you [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) for breaking down ML into a human understandable form
