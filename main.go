package main

import (
	"main/activation"
	"main/layer"
	"main/model"
)

func main() {
	// models.RunFashionModel()

	m := model.Model{}
	m.Add((&layer.Layer{}).Initialization(2, 3))
	m.Add(&layer.DropoutLayer{})
	m.Add(&activation.SoftmaxActivation{})

	provider := model.JSONModelDataProvider{}
	provider.Store("./assets/model.json", &m)
}
