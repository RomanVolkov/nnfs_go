package models

import (
	"log"
	"main/accuracy"
	"main/activation"
	"main/dataset"
	"main/layer"
	"main/loss"
	"main/model"
	"main/optimizer"
)

func RunFashionModel() {
	ds := dataset.FashionMNISTDataset{}
	x, y, err := ds.TrainingDataset()
	if err != nil {
		log.Fatal(err)
	}
	x_val, y_val, err := ds.TestingDataset()
	if err != nil {
		log.Fatal(err)
	}

	_, numInputs := x.Dims()

	m := model.Model{}
	m.Add((&layer.Layer{}).Initialization(numInputs, 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.Layer{}).Initialization(128, 128))
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.Layer{}).Initialization(128, 10))
	m.Add(&activation.SoftmaxActivation{})

	o := optimizer.NewAdam()
	o.Decay = 1e-3

	m.Set(&loss.CategoricalCrossentropyLoss{}, &o, &accuracy.CategorialAccuracy{})
	m.Finalize()
	m.Description()

	trainingData := model.ModelData{X: *x, Y: *y}
	validationData := model.ModelData{X: *x_val, Y: *y_val}
	epochs := 1
	batchSize := 128
	m.Train(trainingData, epochs, &batchSize, 100, &validationData)

	m.Evaluate(validationData, &batchSize)

	dataProvider := model.JSONModelDataProvider{}
	dataProvider.Store("./assets/fashion.json", &m)

	loadedModel, err := dataProvider.Load("./assets/fashion.json")
	if err != nil {
		log.Panic(err)
	}

	loadedModel.Evaluate(validationData, &batchSize)
}
