package models

import (
	"fmt"
	"main/activation"
	"main/classifer"
	"main/dataset"
	"main/layer"
	"main/loss"
	"main/optimizer"

	"gonum.org/v1/gonum/mat"
)

func RunModelV2() {
	x, targetClasses := dataset.SpiralData(1000, 3)
	//	xx := dataset.MockSpiralData()

	dense1 := layer.Layer{}
	dense1.Initialization(2, 64)
	dense1.L2 = layer.Regularizer{Weight: 5e-4, Bias: 5e-4}
	// dense1.Weights = dataset.MockDense1()
	fmt.Println("dense1::weights\n", mat.Formatted(&dense1.Weights))

	activation1 := activation.Activation_ReLU{}

	dropout1 := layer.DropoutLayer{}
	dropout1.Initialization(0.1)

	dense2 := layer.Layer{}
	dense2.Initialization(64, 3)
	// dense2.Weights = dataset.MockDense2()
	fmt.Println("dense2::weights\n", mat.Formatted(&dense2.Weights))

	lossActivation := classifer.ActivationSoftmaxLossCategorialCrossentropy{}

	o := optimizer.NewAdam()
	o.LearningRate = 0.05
	o.CurrentLearningRate = 0.05
	o.Decay = 5e-5
	numberOfEpochs := 10000

	for epoch := 0; epoch < numberOfEpochs; epoch++ {
		dense1.Forward(x)
		activation1.Forward(&dense1.Output)

		dropout1.Forward(&activation1.Output)

		dense2.Forward(&dropout1.Output)

		dataLoss := lossActivation.Forward(&dense2.Output, targetClasses)
		accuracy := loss.CalculateAccuracy(&lossActivation.Output, targetClasses)
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

		lossActivation.Backward(&lossActivation.Output, targetClasses)
		dense2.Backward(&lossActivation.DInputs)
		dropout1.Backward(&dense2.DInputs)
		activation1.Backward(&dropout1.DInputs)
		dense1.Backward(&activation1.DInputs)

		o.PreUpdate()
		o.UpdateParams(&dense1)
		o.UpdateParams(&dense2)
		o.PostUpdate()
	}

	// Testing the model
	testData, testClasses := dataset.SpiralData(100, 3)
	dense1.Forward(testData)
	activation1.Forward(&dense1.Output)
	dense2.Forward(&activation1.Output)

	validationDataLoss := lossActivation.Forward(&dense2.Output, testClasses)
	validationRegularizationLoss := loss.RegularizationLoss(&dense1) + loss.RegularizationLoss(&dense2)
	lossValue := validationDataLoss + validationRegularizationLoss
	testAccuracy := loss.CalculateAccuracy(&lossActivation.Output, testClasses)

	fmt.Println("validation:", "data loss: ", validationDataLoss, " regularization loss: ", validationRegularizationLoss, " loss: ", lossValue, " accuracy: ", testAccuracy)
}
