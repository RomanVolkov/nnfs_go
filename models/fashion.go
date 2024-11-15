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
	"sync"
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
func createCNNBigModel() *model.Model {
	m := &model.Model{}
	m.Name = "CNN - Big"

	inputImageShape := layer.InputShape{Depths: 1, Height: 28, Width: 28}
	cnnLayer1 := (&layer.ConvolutionLayer{}).Initialization(inputImageShape, 32, 3)
	m.Add(cnnLayer1)
	m.Add(&activation.Activation_ReLU{})

	cnnLayer2 := (&layer.ConvolutionLayer{}).Initialization(cnnLayer1.OutputShape, 64, 3)
	m.Add(cnnLayer2)
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DenseLayer{}).Initialization(cnnLayer2.OutputShape.TotalSize(), 250))
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

	wg := sync.WaitGroup{}

	wg.Add(1)
	go func() {
		defer wg.Done()
		trainModeAndStore(createDenseModel(numInputs), "./assets/fashion-dense.json", 10)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		trainModeAndStore(createCNNOneLayerModel(), "./assets/fashion-cnn-1.json", 20)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		trainModeAndStore(createCNNTwoLayerModel(), "./assets/fashion-cnn-2.json", 20)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		trainModeAndStore(createCNNBigModel(), "./assets/fashion-cnn-big.json", 20)
	}()

	wg.Wait()

	fmt.Println("training is done")
}

func LoadModels() {
	fmt.Println("Loading...")
	ds := dataset.FashionMNISTDataset{}
	dataProvider := model.JSONModelDataProvider{}

	cnn1Model, err := dataProvider.Load("./assets/fashion-cnn-1.json")
	if err != nil {
		log.Panic(err)
	}

	cnn2Model, err := dataProvider.Load("./assets/fashion-cnn-2.json")
	if err != nil {
		log.Panic(err)
	}

	cnnBigModel, err := dataProvider.Load("./assets/fashion-cnn-big.json")
	if err != nil {
		log.Panic(err)
	}

	denseModel, err := dataProvider.Load("./assets/fashion-dense.json")
	if err != nil {
		log.Panic(err)
	}

	batchSize := 128
	x_val, y_val, err := ds.TestingDataset()
	if err != nil {
		log.Fatal(err)
	}
	validationData := model.ModelData{X: *x_val, Y: *y_val}

	fmt.Println()
	fmt.Println("CNN-1")
	cnn1Model.Evaluate(validationData, &batchSize)

	fmt.Println()
	fmt.Println("CNN-2")
	cnn2Model.Evaluate(validationData, &batchSize)

	fmt.Println()
	fmt.Println("CNN-Big")
	cnnBigModel.Evaluate(validationData, &batchSize)

	fmt.Println()
	fmt.Println("Dense")
	denseModel.Evaluate(validationData, &batchSize)
}
