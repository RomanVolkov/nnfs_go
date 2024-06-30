package main

import (
	"main/activation"
	"main/dataset"
	"main/layer"
	"main/loss"
	"main/model"
	"main/optimizer"
)

func main() {
	// models.RunBinaryModel()
	// RunModelV2()
	// RunBinaryModel()
	// models.RunRegressionModel()
	x, y := dataset.SineData(1000)

	model := model.Model{}

	model.Add((&layer.Layer{}).Initialization(1, 64))
	model.Add(&activation.Activation_ReLU{})
	model.Add((&layer.Layer{}).Initialization(64, 64))
	model.Add(&activation.Activation_ReLU{})
	model.Add((&layer.Layer{}).Initialization(64, 1))
	model.Add(&activation.LinearActivation{})

	lossF := loss.MeanSquaredErrorLoss{}
	o := optimizer.NewAdam()
	o.LearningRate = 0.005
	o.CurrentLearningRate = 0.005
	o.Decay = 1e-3

	model.Set(&lossF, &o)
	model.Description()

	model.Finalize()

	model.Train(x, y, 10000, 100)
}
