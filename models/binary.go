package models

import (
	"main/accuracy"
	"main/activation"
	"main/dataset"
	"main/layer"
	"main/loss"
	"main/model"
	"main/optimizer"
)

func RunBinaryModel() {
	x, y := dataset.SpiralData(100, 2)
	x_val, y_val := dataset.SpiralData(100, 2)

	m := model.Model{}

	layer1 := layer.Layer{}
	layer1.Initialization(2, 64)
	layer1.L2 = layer.Regularizer{Weight: 5e-4, Bias: 5e-4}
	m.Add(&layer1)
	m.Add(&activation.Activation_ReLU{})
	m.Add((&layer.Layer{}).Initialization(64, 1))
	m.Add(&activation.SigmoidActivation{})

	l := loss.BinaryCrossentropyLoss{}
	o := optimizer.NewAdam()
	o.Decay = 5e-7
	a := accuracy.BinaryCategorialAccuracy{}

	m.Set(&l, &o, &a)

	m.Finalize()
	m.Train(model.ModelData{X: x, Y: y}, 10000, 100, &model.ModelData{X: x_val, Y: y_val})
}
