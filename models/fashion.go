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

	batchSize := 128
	m.Description()
	m.Train(model.ModelData{X: *x, Y: *y}, 10, &batchSize, 100, &model.ModelData{X: *x_val, Y: *y_val})
}
