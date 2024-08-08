# Neural networks from Scratch in Go


Here I study the book http://nnfs.io and implement all the code with Golang instead of Python. The aim is to learn how neural networks are made inside and how it looks like to build is fully from the scratch without using frameworks like pytorch or tensorflow. You can find here implementation of Dense Layer, several common accuracy functions, several common loss functions, common optimizers and simple types of NN modes like regression or classification. 

All code is written in learning purposes for dive deeper into machine learning topics and Golang.

Implementation of this repo was recorded and published on YouTube, [here](https://www.youtube.com/playlist?list=PLzDkoEk_dpxoXlTDOIeDPxl83PprF2vKW) is the link to playlist with all the videos.

### Build&run

Just use 
- `make build` to build the project
- `make run` to run the project

In order to train a classification model using Fashion MNIST dataset you have to unzip `assets/fashion_mnist_images.zip` into `assets/fashion_mnist_images` and then train it, but my trained version is already in the repo `fashion.json`
