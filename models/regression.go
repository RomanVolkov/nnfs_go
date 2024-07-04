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

func RunRegressionModel() {
	x, y := dataset.SineData(1000)

	m := model.Model{}

	m.Add((&layer.Layer{}).Initialization(1, 64))
	m.Add(&activation.Activation_ReLU{})
	m.Add((&layer.Layer{}).Initialization(64, 64))
	m.Add(&activation.Activation_ReLU{})
	m.Add((&layer.Layer{}).Initialization(64, 1))
	m.Add(&activation.LinearActivation{})

	lossF := loss.MeanSquaredErrorLoss{}
	o := optimizer.NewAdam()
	o.LearningRate = 0.005
	o.CurrentLearningRate = 0.005
	o.Decay = 1e-3

	accuracy := accuracy.RegressionAccuracy{}

	m.Set(&lossF, &o, &accuracy)
	m.Description()

	m.Finalize()

	m.Train(model.ModelData{X: x, Y: y}, 10000, 100, nil)
}
