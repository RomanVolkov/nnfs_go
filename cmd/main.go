package main

import (
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

	model := model.Model{}

	model.Add((&layer.Layer{}).Initialization(10, 10))
	lossF := loss.MeanAbsoluteErrorLoss{}
	o := optimizer.NewAdam()

	model.Set(&lossF, &o)
	model.Description()
}
