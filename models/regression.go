package models

import (
	"fmt"
	"main/activation"
	"main/dataset"
	"main/layer"
	"main/loss"
	"main/optimizer"

	"gonum.org/v1/gonum/mat"
)

func RunRegressionModel() {
	x, y := dataset.SineData(1000)
	data := mat.NewDense(1000, 1, x.RawMatrix().Data)

	dense1 := layer.Layer{}
	dense1.Initialization(1, 64)
	activation1 := activation.Activation_ReLU{}

	dense2 := layer.Layer{}
	dense2.Initialization(64, 64)
	activation2 := activation.Activation_ReLU{}

	dense3 := layer.Layer{}
	dense3.Initialization(64, 1)
	activation3 := activation.LinearActivation{}

	lossFunction := loss.MeanSquaredErrorLoss{}

	optimizer := optimizer.NewAdam()
	optimizer.LearningRate = 0.001
	optimizer.CurrentLearningRate = 0.001
	optimizer.Decay = 1e-3

	numberOfEpochs := 10001

	for epoch := 0; epoch < numberOfEpochs; epoch++ {
		dense1.Forward(data)
		activation1.Forward(&dense1.Output)

		dense2.Forward(&activation1.Output)
		activation2.Forward(&dense2.Output)

		dense3.Forward(&activation2.Output)
		activation3.Forward(&dense3.Output)

		dataLoss := lossFunction.Calculate(&activation3.Output, y)
		regularizationLoss := loss.RegularizationLoss(&dense1) + loss.RegularizationLoss(&dense2) + loss.RegularizationLoss(&dense3)
		lossValue := dataLoss + regularizationLoss

		accuracy := loss.CalculateRegressionAccuracy(&activation3.Output, y)

		if epoch%100 == 0 {
			fmt.Println("====================")
			fmt.Println("epoch: ", epoch)
			fmt.Println("data loss: ", dataLoss)
			fmt.Println("regularization loss: ", regularizationLoss)
			fmt.Println("loss: ", lossValue)
			fmt.Println("accuracy: ", accuracy)
			fmt.Println("learning rate: ", optimizer.CurrentLearningRate)
		}

		lossFunction.Backward(&activation3.Output, y)
		activation3.Backward(&lossFunction.DInputs)
		dense3.Backward(&activation3.DInputs)
		activation2.Backward(&dense3.DInputs)
		dense2.Backward(&activation2.DInputs)
		activation1.Backward(&dense2.DInputs)
		dense1.Backward(&activation1.DInputs)

		optimizer.PreUpdate()
		optimizer.UpdateParams(&dense1)
		optimizer.UpdateParams(&dense2)
		optimizer.UpdateParams(&dense3)
		optimizer.PostUpdate()
	}

	fmt.Println("Testing run")
	xTest, _ := dataset.SineData(1000)
	testData := mat.NewDense(1000, 1, xTest)

	dense1.Forward(testData)
	activation1.Forward(&dense1.Output)

	dense2.Forward(&activation1.Output)
	activation2.Forward(&dense2.Output)

	dense3.Forward(&activation2.Output)
	activation3.Forward(&dense3.Output)

	xx := make([][]float64, 1)
	xx[0] = xTest

	yy := make([][]float64, 1)
	yy[0] = activation3.Output.RawMatrix().Data

	dataset.ScatterSineData(xx, yy)
}
