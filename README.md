# Neural networks from Scratch in Go


Here I study the book http://nnfs.io and implement all the code with Golang instead of Python. The aim is to learn how neural networks are made inside and how it looks like to build is fully from the scratch without using frameworks like pytorch or tensorflow. You can find here implementation of Dense Layer, Convolutions and others, several common accuracy functions, several common loss functions, common optimizers and simple types of NN modes like regression or classification.

All code is written in learning purposes for dive deeper into machine learning topics and Golang. There is no intention to build yet another deep learning tool, this is why the code mostly focused on readability with some small level of performance-focus.

Implementation of this repo was recorded and published on YouTube, [here](https://www.youtube.com/playlist?list=PLzDkoEk_dpxoXlTDOIeDPxl83PprF2vKW) is the link to playlist with all the videos.

Second version of learning was focused on build a CNN and there is another video series [here](https://www.youtube.com/playlist?list=PLzDkoEk_dpxoP4dMzYxoK_u2PZ2KMbXH7)

### Build&run

Just use 
- `make build` to build the project
- `make run` to run the project. You can define which models you want to use inside `main.go` -> `main` func. You can take a look to `models` package to see what are the examples.
- `make serve` to run simple http-server to use Fashion MNIST model

In order to train a classification model using Fashion MNIST dataset you have to unzip `assets/fashion_mnist_images.zip` into `assets/fashion_mnist_images` and then train it, but many already trained models are stored in `assets/` folder.
