# Summary

In this workshop, we covered:

```{admonition} 1. Understand the fundamentals of machine learning and deep learning.

- [x] _Machine learning and deep learning are a range of prediction methods that learn associations from training data._
- [x] _The objective is for the models to generalise to new data._
- [x] _They mainly use tensors (multi-dimensional arrays) as inputs._
- [x] _Problems are mainly either supervised (if you provide labels) or unsupervised (if you don't provide labels)._
- [x] _Problems are either classification (if you're trying to predict a discrete category) or regression (if you're trying to predict a continuous number)._
- [x] _Data is split into training, validation, and test sets._
- [x] _The models only learn from the training data._
- [x] _The test set is used only once._
- [x] _Hyperparameters are set before model training._
- [x] _Parameters (i.e., the weights and biases) are learnt during model training._
- [x] _The aim is to minimise the loss function._
- [x] _The model underfits when it has high bias._
- [x] _The model overfits when it has high variance._

```


```{admonition} 2. Know how to use key tools, including:

- [x] [scikit-learn](https://scikit-learn.org/stable/)
    - [x] _scikit-learn is great for classic machine learning problems._
- [x] [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)
    - [x] _TensorFlow is great for deep learning problems._
    - [x] _Keras (high-level API for TensorFlow) has many high-level objects to help you create deep learning models._
- [x] [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/)
    - [x] _PyTorch is great for deep learning problems._
    - [x] _PyTorch Lightning (high-level API for PyTorch) has many high-level objects to help you create deep learning models._
- [x] _You can use low-level APIs for any custom objects._
- [x] _Explore your data before using it._
- [x] _Check your model before fitting the training data to it._
- [x] _Evaluate your model and analyse the errors it makes._    

```


```{admonition} 3. Be aware of good practices for data, such as pipelines and modules.

- [x] _Always split the data into train and test subsets first, before any pre-processing._
- [x] _Never fit to the test data._
- [x] _Use a data pipeline._
- [x] _Use a random seed and any available deterministic functionalities for reproducibility._
    - [x] _Try and reproduce your own work, to check that it is reproducible._
- [x] _Consider optimising the data pipeline with:_
    - [x] _Shuffling._
    - [x] _Batching._
    - [x] _Caching._
    - [x] _Prefetching._
    - [x] _Parallel data extraction._
    - [x] _Data augmentation._
    - [x] _Parallel data transformation._
    - [x] _Vectorised mapping._
    - [x] _Mixed precision._

```


```{admonition} 4. Be aware of good practices for models, such as hyperparameter tuning, transfer learning, and callbacks.

- [x] _Tune hyperparamaters for the best model fit._
- [x] _Use transfer learning to save computation on similar problems._
- [x] _Consider using callbacks to help with model training, such as:_
    - [x] _Checkpoints._
    - [x] _Fault tolerance._
    - [x] _Logging._
    - [x] _Profiling._
    - [x] _Early stopping._
    - [x] _Learning rate decay._

```

```{admonition} 5. Be able to undertake distributed training.

- [x] _Ensure that you really need to use distributed devices._
- [x] _Check everything first works on a single device._
- [x] _Use data parallelism (to split the data over multiple devices)._
- [x] _Take care when setting the global batch size._
- [x] _Check the efficiency of your jobs to ensure utilising the requested resources._
- [x] _When moving from Jupyter to HPC:_
    - [x] _Clean non-essential code._
    - [x] _Refactor Jupyter Notebook code into functions._
    - [x] _Create a Python script._
    - [x] _Create submission script._
    - [x] _Create unit tests._

```


## Next steps

For things that you're interested in:

- Try things out yourself (e.g., play around with the examples).
- Check out the {ref}`Online Courses <online_courses>`.
