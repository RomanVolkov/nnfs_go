package models

import (
	"main/accuracy"
	"main/activation"
	"main/dataset"
	"main/layer"
	"main/model"
	"main/optimizations"
	"main/optimizer"
)

func RunCategorialModel() {
	x, y := dataset.SpiralData(1000, 3)
	x_val, y_val := dataset.SpiralData(1000, 3)

	m := model.Model{}

	layer1 := layer.Layer{}
	layer1.Initialization(2, 512)
	layer1.L2 = layer.Regularizer{Weight: 5e-4, Bias: 5e-4}
	m.Add(&layer1)
	m.Add(&activation.Activation_ReLU{})

	m.Add((&layer.DropoutLayer{}).Initialization(0.1))

	m.Add((&layer.Layer{}).Initialization(512, 3))

	activ, l := optimizations.MakeOptimizedCategorialCrossentropy()
	m.Add(&activ)

	o := optimizer.NewAdam()
	o.LearningRate = 0.05
	o.CurrentLearningRate = 0.05
	o.Decay = 5e-5
	a := accuracy.CategorialAccuracy{}

	m.Set(&l, &o, &a)

	m.Finalize()
	m.Train(model.ModelData{X: x, Y: y}, 10000, 100, &model.ModelData{X: x_val, Y: y_val})
}
