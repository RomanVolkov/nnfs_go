package main

import (
	"main/accuracy"
	"main/activation"
	"main/layer"
	"main/loss"
	"main/model"
	"main/optimizer"
)

func main() {
	// models.RunFashionModel()

	m := model.Model{}
	m.Add((&layer.Layer{}).Initialization(2, 3))
	m.Add(&layer.DropoutLayer{})
	m.Add(&activation.SoftmaxActivation{})

	m.Set(&loss.MeanSquaredErrorLoss{}, &optimizer.OptimizerAdam{}, &accuracy.RegressionAccuracy{})

	provider := model.JSONModelDataProvider{}
	provider.Store("./assets/model.json", &m)
}
