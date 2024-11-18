package models

import (
	"errors"
	"fmt"
	"log"
	"main/accuracy"
	"main/activation"
	"main/dataset"
	"main/layer"
	"main/loss"
	"main/model"
	"main/optimizer"
	"os"
)

func createDenseModel(numInputs int) *model.Model {
	m := &model.Model{}
	m.Name = "Dense Model"
	m.Add((&layer.DenseLayer{}).Initialization(numInputs, 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(128, 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(128, 10))
	m.Add(&activation.SoftmaxActivation{})

	o := optimizer.NewAdam()
	o.Decay = 1e-3

	m.Set(&loss.CategoricalCrossentropyLoss{}, &o, &accuracy.CategorialAccuracy{})
	m.Finalize()

	return m
}

func createCNNOneLayerModel() *model.Model {
	m := &model.Model{}
	m.Name = "CNN - 1"

	inputImageShape := layer.InputShape{Depths: 1, Height: 28, Width: 28}
	cnnLayer := (&layer.ConvolutionLayer{}).Initialization(inputImageShape, 3, 5)
	m.Add(cnnLayer)
	m.Add(&activation.SigmoidActivation{})

	m.Add((&layer.DenseLayer{}).Initialization(cnnLayer.OutputShape.TotalSize(), 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(128, 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(128, 10))
	m.Add(&activation.SoftmaxActivation{})

	// o := optimizer.NewSGD(0.5, 1e-3, 0)
	o := optimizer.NewAdam()
	o.Decay = 1e-5

	m.Set(&loss.CategoricalCrossentropyLoss{}, &o, &accuracy.CategorialAccuracy{})
	m.Finalize()
	return m
}

func createCNNWithMaxPoolingLayerModel() *model.Model {
	m := &model.Model{}
	m.Name = "CNN - MaxPooling"

	inputImageShape := layer.InputShape{Depths: 1, Height: 28, Width: 28}
	cnnLayer := (&layer.ConvolutionLayer{}).Initialization(inputImageShape, 3, 5)
	m.Add(cnnLayer)
	m.Add(&activation.SigmoidActivation{})
	maxPooling := (&layer.MaxPoolingLayer{}).Initialization(cnnLayer.OutputShape, 2)
	m.Add(maxPooling)

	m.Add((&layer.DenseLayer{}).Initialization(maxPooling.OutputShape.TotalSize(), 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(128, 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(128, 10))
	m.Add(&activation.SoftmaxActivation{})

	// o := optimizer.NewSGD(0.5, 1e-3, 0)
	o := optimizer.NewAdam()
	o.Decay = 1e-5

	m.Set(&loss.CategoricalCrossentropyLoss{}, &o, &accuracy.CategorialAccuracy{})
	m.Finalize()
	return m
}

func createCNNTwoLayerModel() *model.Model {
	m := &model.Model{}
	m.Name = "CNN - 2"

	inputImageShape := layer.InputShape{Depths: 1, Height: 28, Width: 28}
	cnnLayer1 := (&layer.ConvolutionLayer{}).Initialization(inputImageShape, 3, 5)
	m.Add(cnnLayer1)
	m.Add(&activation.SigmoidActivation{})

	cnnLayer2 := (&layer.ConvolutionLayer{}).Initialization(cnnLayer1.OutputShape, 2, 3)
	m.Add(cnnLayer2)
	m.Add(&activation.SigmoidActivation{})

	m.Add((&layer.DenseLayer{}).Initialization(cnnLayer2.OutputShape.TotalSize(), 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(128, 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(128, 10))
	m.Add(&activation.SoftmaxActivation{})

	// o := optimizer.NewSGD(0.5, 1e-3, 0)
	o := optimizer.NewAdam()
	o.Decay = 1e-5

	m.Set(&loss.CategoricalCrossentropyLoss{}, &o, &accuracy.CategorialAccuracy{})
	m.Finalize()
	return m
}

// https://github.com/guilhermedom/cnn-fashion-mnist/blob/main/notebooks/1.0-gdfs-cnn-fashion-mnist.ipynb
// cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
// cnn_model.add(layers.MaxPool2D(pool_size=(2, 2)))
// cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
// cnn_model.add(layers.MaxPool2D(pool_size=(2, 2)))
// cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
// cnn_model.add(layers.Flatten())
// cnn_model.add(layers.Dense(250, activation='relu'))
// cnn_model.add(layers.Dense(125, activation='relu'))
// cnn_model.add(layers.Dense(60, activation='relu'))
// cnn_model.add(layers.Dense(10, activation='softmax'))

func createCNNBigModel() *model.Model {
	m := &model.Model{}
	m.Name = "CNN - Big"

	inputImageShape := layer.InputShape{Depths: 1, Height: 28, Width: 28}
	cnnLayer1 := (&layer.ConvolutionLayer{}).Initialization(inputImageShape, 32, 3)
	m.Add(cnnLayer1)
	m.Add(&activation.Activation_ReLU{})
	maxPooling1 := (&layer.MaxPoolingLayer{}).Initialization(cnnLayer1.OutputShape, 2)
	m.Add(maxPooling1)

	cnnLayer2 := (&layer.ConvolutionLayer{}).Initialization(maxPooling1.OutputShape, 64, 3)
	m.Add(cnnLayer2)
	m.Add(&activation.Activation_ReLU{})
	maxPooling2 := (&layer.MaxPoolingLayer{}).Initialization(cnnLayer2.OutputShape, 2)
	m.Add(maxPooling2)

	// cnnLayer3 := (&layer.ConvolutionLayer{}).Initialization(maxPooling2.OutputShape, 64, 3)
	// m.Add(cnnLayer3)
	// m.Add(&activation.Activation_ReLU{})
	// maxPooling3 := (&layer.MaxPoolingLayer{}).Initialization(cnnLayer3.OutputShape, 2)
	// m.Add(maxPooling3)

	m.Add((&layer.DenseLayer{}).Initialization(maxPooling2.OutputShape.TotalSize(), 250))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(250, 125))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(125, 60))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(60, 10))
	m.Add(&activation.SoftmaxActivation{})

	// o := optimizer.NewSGD(0.5, 1e-3, 0)
	o := optimizer.NewAdam()
	o.Decay = 1e-5

	m.Set(&loss.CategoricalCrossentropyLoss{}, &o, &accuracy.CategorialAccuracy{})
	m.Finalize()
	return m
}

// Loads FashionMNISTDataset and runs Train process for input model
func trainModeAndStore(m *model.Model, path string, epochs int) error {
	if m == nil {
		return errors.New("Missing model")
	}

	ds := dataset.FashionMNISTDataset{}
	x, y, err := ds.TrainingDataset()
	if err != nil {
		log.Fatal(err)
	}
	x_val, y_val, err := ds.TestingDataset()
	if err != nil {
		log.Fatal(err)
	}

	m.Description()

	trainingData := model.ModelData{X: *x, Y: *y}
	validationData := model.ModelData{X: *x_val, Y: *y_val}

	batchSize := 128
	m.Train(trainingData, epochs, &batchSize, 100, &validationData)
	m.Evaluate(validationData, &batchSize)

	dataProvider := model.JSONModelDataProvider{}
	err = dataProvider.Store(path, m)
	if err != nil {
		return err
	}

	return nil
}

// so the idea that since current model training runs in one thread I can spawn several
// models to train all at once. and then compare results
func TrainModels() {
	ds := dataset.FashionMNISTDataset{}
	x, _, err := ds.TrainingDataset()
	if err != nil {
		log.Fatal(err)
	}
	_, numInputs := x.Dims()

	trainModeAndStore(createDenseModel(numInputs), "./assets/fashion-dense.json", 10)
	trainModeAndStore(createCNNOneLayerModel(), "./assets/fashion-cnn-1.json", 30)
	trainModeAndStore(createCNNWithMaxPoolingLayerModel(), "./assets/fashion-cnn-max-pooling.json", 30)
	trainModeAndStore(createCNNTwoLayerModel(), "./assets/fashion-cnn-2.json", 30)
	// trainModeAndStore(createCNNBigModel(), "./assets/fashion-cnn-big.json", 20)

	fmt.Println("training is done")
}

func LoadModels() {
	fmt.Println("Loading...")
	ds := dataset.FashionMNISTDataset{}
	dataProvider := model.JSONModelDataProvider{}

	mArray := []string{
		"./assets/fashion-cnn-1.json",
		"./assets/fashion-cnn-max-pooling.json",
		"./assets/fashion-cnn-2.json",
		"./assets/fashion-cnn-big.json",
		"./assets/fashion-dense.json",
	}

	batchSize := 128
	x_val, y_val, err := ds.TestingDataset()
	if err != nil {
		log.Fatal(err)
	}
	validationData := model.ModelData{X: *x_val, Y: *y_val}

	for _, path := range mArray {
		fmt.Println()
		if _, err := os.Stat(path); os.IsNotExist(err) {
			fmt.Println()
			fmt.Println("Missing model:", path)
			fmt.Println()
			continue
		}

		m, err := dataProvider.Load(path)
		if err != nil {
			log.Panic(err)
		}

		fmt.Println(m.Name)
		m.Evaluate(validationData, &batchSize)
	}
}
