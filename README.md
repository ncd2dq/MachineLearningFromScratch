# Machine Learning From Scratch

This repo is intended to give you a good understanding of the basics of Machine Learning
from scratch (no dependencies except numpy). 

There are many amazing libraries out there like TensorFlow, pytorch, and scikitlearn - just to name a few - that use GPU optimization, implement the latest algorithms, and are written in C to deliver the best performance. 
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
pip install numpy
```

For MacOS
```
python -m venv venv
. venv/bin/activate
pip install numpy
```

## Contents

* Supervised Learning
	* Neural Networks
		* What is it?
			* Different Types [Chart](https://cdn-images-1.medium.com/max/1000/1*cuTSPlTq0a_327iTPJyD-Q.png)
		* Build it
	* Decision Trees
		* What is it?
		* Build it
	* Linear Regression
		* What is it?
		* Build it
	* Support Vector Machines
		* What is it?
		* Build it
	* Decision Trees
		* What is it?
		* Build it

* Unsupervised Learning
	* k-means clustering
		* What is it?
		* Build it

* Reinforcement Learning
	* Q-Learning
		* What is it?
		* Build it
	* Deep Q-Learning (Deep Reinforcement Learning)
		* What is it?
		* Build it

* Evolutionary Algorithms
	* Evolutionary Algorithms
		* What is it?
		* Build it
	* Neuro-Evolutionary Network (Neural Networks + Evolutionary Algorithms)
		* What is it?
		* Build it

* Machine Learning Terminology

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Nicholas Dima** - *Tutorials/Code* - [Me](www.nickdima.com)

## License

This project is licensed under the MIT License

## Acknowledgments

* The blog post that got me started with ML and inspired this tutorial series - [I am trask](https://iamtrask.github.io/2015/07/12/basic-python-network/)
* Thank you [Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) for your dedication to ML videos
* Thank you [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) for breaking down ML into a human understandable form
