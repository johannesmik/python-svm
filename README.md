This was a project I did for the Machine Learning course at Technical University Munich in 2014.

** This project is not maintained **

For this project I wrote a Support Vector Machine in Python from scratch, and then used it to detect letters in the [Letter Image Recognition Dataset](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition). Result:  After training the SVM using a 10-fold cross-validation, the resulting error rate was 6.5% on the testing data.

# Dependencies

## Python 2.7, Numpy and Matplotlib

On Ubuntu those might be installed using:

```
$ sudo apt-get install python python-matplotlib python-numpy
```

## Letter Recognition Dataset
  
Also download the [Letter Recognition](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition) data and put the `letter-recognition.data` file into a directory called `data`:

```
$ mkdir data
$ curl http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data -o ./data/letter-recognition.data
```

# Running

For plotting a small example dataset using One-vs-One:

```
$ python predict.py 2d_examples
```

For plotting a small example dataset using One-vs-All:

```
$ python predict.py 2d_examples_oneall
```

For finding meta-parameters sigma / c (may take some time)

```
$ python predict.py find_sigma
$ python predict.py find_c
```

For training on the full letter recognition dataset, One-vs-One. Please download the 

```
$ python predict.py letter_data
```

For training on the first 7000 samples of letter recognition dataset, One-vs-All:

```
$ python predict.py letter_data_oneall
```

# References

 - Peter W Frey and David J Slate. Letter recognition using holland-style adaptive classifiers. Machine Learning, 6:161, 1991.
 - John C Platt. Fast training of support vector machines using sequential minimal optimization. Advances in Kernel Methods - Support Vector Learning, 1999.
 - John C Platt. Using analytic qp and sparseness to speed training of support vector machines. Advances in neural information processing systems, pages 557–563, 1999.
