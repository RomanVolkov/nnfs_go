package main

import (
	"fmt"
	"main/activation"
	"main/classifer"
	"main/dataset"
	"main/layer"
	"main/loss"
	"main/optimizer"
	"time"

	"gonum.org/v1/gonum/mat"
)

func RunModel() {
	inputs, targetClasses := dataset.SpiralData(100, 3)

	fmt.Println("Layers")
	dense1 := layer.Layer{}
	dense1.Initialization(2, 3)
	dense1.Forward(inputs)
	activation1 := activation.Activation_ReLU{}
	activation1.Forward(&dense1.Output)
	fmt.Println(mat.Formatted(&activation1.Output))

	dense2 := layer.Layer{}
	dense2.Initialization(3, 3)
	dense2.Forward(&activation1.Output)

	lossActivation := classifer.ActivationSoftmaxLossCategorialCrossentropy{}
	lossValue := lossActivation.Forward(&dense2.Output, targetClasses)

	accuracy := loss.CalculateAccuracy(&lossActivation.Output, targetClasses)
	fmt.Println("loss: ", lossValue)
	fmt.Println("accuracy: ", accuracy)

	fmt.Println("==================")
	fmt.Println("Backward pass")
	lossActivation.Backward(&lossActivation.Output, targetClasses)
	dense2.Backward(&lossActivation.DInputs)
	activation1.Backward(&dense2.DInputs)
	dense1.Backward(&activation1.DInputs)

	fmt.Println(mat.Formatted(&dense1.DWeights))
	fmt.Println(mat.Formatted(&dense1.DBiases))
	fmt.Println(mat.Formatted(&dense2.DWeights))
	fmt.Println(mat.Formatted(&dense2.DBiases))

	o := optimizer.NewSGD()
	o.UpdateParams(&dense1)
	o.UpdateParams(&dense2)
}

func testBackwardPass() {
	// dvalues := mat.NewDense(3, 3, []float64{1., 1., 1., 2., 2., 2., 3., 3., 3.})
	inputs := mat.NewDense(3, 4, []float64{1, 2, 3, 2.5, 2., 5., -1., 2, -1.5, 2.7, 3.3, -0.8})
	// storing weights directly as T()
	weights := mat.NewDense(4, 3, []float64{0.2, 0.5, -0.26, 0.8, -0.91, -0.27, -0.5, 0.26, 0.17, 1., -0.5, 0.87})
	biases := mat.NewDense(1, 3, []float64{2, 3, 0.5})
	layer := layer.Layer{}
	layer.Weights = *weights
	layer.Biases = *biases

	layer.Forward(inputs)
	r, c := layer.Output.Dims()
	fmt.Println(r, c)

	activation := activation.Activation_ReLU{}
	activation.Forward(&layer.Output)

	dvalues := mat.DenseCopyOf(&activation.Output)
	activation.Backward(dvalues)
	r, c = activation.DInputs.Dims()
	fmt.Println(r, c)
	layer.Backward(&activation.DInputs)
}

func testBackpropagate() {
	classTargets := []uint8{0, 1, 1}
	softmaxOutputs := mat.NewDense(3, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08})

	start1 := time.Now()
	for i := 0; i < 10000; i++ {
		softmaxLoss := classifer.ActivationSoftmaxLossCategorialCrossentropy{}
		softmaxLoss.Initialize()
		softmaxLoss.Backward(softmaxOutputs, classTargets)
	}
	duration1 := time.Since(start1)

	start2 := time.Now()
	for i := 0; i < 10000; i++ {
		activation1 := activation.SoftmaxActivation{}
		activation1.Output = *softmaxOutputs
		loss1 := loss.CategoricalCrossentropyLoss{}
		loss1.Backward(*softmaxOutputs, classTargets)
		activation1.Backward(&loss1.DInputs)
	}
	duration2 := time.Since(start2)

	fmt.Println(duration2.Seconds() / duration1.Seconds())
}

func dumpDense(values mat.Dense) {
	str := ""
	for _, value := range values.RawMatrix().Data {
		str = str + fmt.Sprintf("%f,", value)
	}
	fmt.Println(str)
}

func RunModelV2() {
	_, targetClasses := dataset.SpiralData(100, 3)
	xx := dataset.MockSpiralData()

	dense1 := layer.Layer{}
	dense1.Initialization(2, 10)
	dense1.Weights = dataset.MockDense1()
	fmt.Println("dense1::weights\n", mat.Formatted(&dense1.Weights))

	activation1 := activation.Activation_ReLU{}

	dense2 := layer.Layer{}
	dense2.Initialization(10, 3)
	dense2.Weights = dataset.MockDense2()
	fmt.Println("dense2::weights\n", mat.Formatted(&dense2.Weights))

	lossActivation := classifer.ActivationSoftmaxLossCategorialCrossentropy{}

	o := optimizer.NewSGD()
	numberOfEpochs := 10000

	for epoch := 0; epoch < numberOfEpochs; epoch++ {
		dense1.Forward(&xx)
		activation1.Forward(&dense1.Output)
		dense2.Forward(&activation1.Output)
		lossValue := lossActivation.Forward(&dense2.Output, targetClasses)
		accuracy := loss.CalculateAccuracy(&lossActivation.Output, targetClasses)

		if epoch%100 == 0 {
			fmt.Println("epoch: ", epoch)
			fmt.Println("loss: ", lossValue)
			fmt.Println("accuracy: ", accuracy)
		}

		lossActivation.Backward(&lossActivation.Output, targetClasses)
		dense2.Backward(&lossActivation.DInputs)
		activation1.Backward(&dense2.DInputs)
		dense1.Backward(&activation1.DInputs)

		o.UpdateParams(&dense1)
		o.UpdateParams(&dense2)
	}
}

func main() {
	// RunModel()
	// RunSimpleModel()
	// testBackwardPass()
	// testBackpropagate()
	RunModelV2()
}
