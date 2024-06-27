package models

import (
	"fmt"
	"main/activation"
	"main/dataset"
	"main/layer"
	"main/loss"
	"main/optimizer"
)

func RunBinaryModel() {
	x, targetClasses := dataset.SpiralData(100, 2)
	// x = dataset.MockSpiralData()

	dense1 := layer.Layer{}
	dense1.Initialization(2, 64)
	// dense1.Weights = *dataset.MockDense1()
	dense1.L2 = layer.Regularizer{Weight: 5e-4, Bias: 5e-4}

	activation1 := activation.Activation_ReLU{}

	dense2 := layer.Layer{}
	dense2.Initialization(64, 1)
	// dense2.Weights = *dataset.MockDense2()

	activation2 := activation.SigmoidActivation{}
	lossFunction := loss.BinaryCrossentropyLoss{}

	o := optimizer.NewAdam()
	o.Decay = 5e-7
	numberOfEpochs := 10001
	// numberOfEpochs := 1

	for epoch := 0; epoch < numberOfEpochs; epoch++ {
		dense1.Forward(x)
		activation1.Forward(&dense1.Output)

		dense2.Forward(&activation1.Output)
		activation2.Forward(&dense2.Output)

		accuracy := loss.CalculateBinaryAccuracy(&activation2.Output, targetClasses)

		dataLoss := lossFunction.Calculate(&activation2.Output, targetClasses)
		regularizationLoss := loss.RegularizationLoss(&dense1) + loss.RegularizationLoss(&dense2)
		lossValue := dataLoss + regularizationLoss

		if epoch%100 == 0 {
			fmt.Println("epoch: ", epoch)
			fmt.Println("data loss: ", dataLoss)
			fmt.Println("regularization loss: ", regularizationLoss)
			fmt.Println("loss: ", lossValue)
			fmt.Println("accuracy: ", accuracy)
			fmt.Println("learning rate: ", o.CurrentLearningRate)
		}

		lossFunction.Backward(&activation2.Output, targetClasses)
		activation2.Backward(&lossFunction.DInputs)
		dense2.Backward(&activation2.DInputs)
		activation1.Backward(&dense2.DInputs)
		dense1.Backward(&activation1.DInputs)

		o.PreUpdate()
		o.UpdateParams(&dense1)
		o.UpdateParams(&dense2)
		o.PostUpdate()
	}

	// Testing the model
	testData, testClasses := dataset.SpiralData(100, 2)
	dense1.Forward(testData)
	activation1.Forward(&dense1.Output)
	dense2.Forward(&activation1.Output)
	activation2.Forward(&dense2.Output)

	testAccuracy := loss.CalculateBinaryAccuracy(&activation2.Output, testClasses)

	validationDataLoss := lossFunction.Calculate(&activation2.Output, testClasses)
	validationRegularizationLoss := loss.RegularizationLoss(&dense1) + loss.RegularizationLoss(&dense2)

	validationLoss := validationDataLoss + validationRegularizationLoss

	fmt.Println("validation:", "data loss: ", validationDataLoss, " regularization loss: ", validationRegularizationLoss, " loss: ", validationLoss, " accuracy: ", testAccuracy)
}
