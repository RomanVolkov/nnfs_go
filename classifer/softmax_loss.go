package classifer

import (
	"main/activation"
	"main/loss"

	"gonum.org/v1/gonum/mat"
)

type ActivationSoftmaxLossCategorialCrossentropy struct {
	Output     mat.Dense
	activation activation.SoftmaxActivation
	loss       loss.CategoricalCrossentropyLoss
	DInputs    mat.Dense
}

func (a *ActivationSoftmaxLossCategorialCrossentropy) Initialize() {
	a.activation = activation.SoftmaxActivation{}
	a.loss = loss.CategoricalCrossentropyLoss{}
}

func (a *ActivationSoftmaxLossCategorialCrossentropy) Forward(inputs *mat.Dense, y []uint8) float64 {
	a.activation.Forward(inputs)
	a.Output = *mat.DenseCopyOf(&a.activation.Output)
	return a.loss.Calculate(&a.Output, y)
}

func (a *ActivationSoftmaxLossCategorialCrossentropy) Backward(dvalues *mat.Dense, y []uint8) {
	samplesCount, _ := dvalues.Dims()
	a.DInputs = *mat.DenseCopyOf(dvalues)

	for i := 0; i < samplesCount; i++ {
		value := a.DInputs.At(i, int(y[i])) - 1.0
		a.DInputs.Set(i, int(y[i]), value)
	}

	a.DInputs.Apply(func(i, j int, v float64) float64 {
		return v / float64(samplesCount)
	}, &a.DInputs)
}
