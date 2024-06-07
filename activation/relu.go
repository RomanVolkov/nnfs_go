package activation

import (
	"gonum.org/v1/gonum/mat"
)

type Activation_ReLU struct {
	inputs  mat.Dense
	DInputs mat.Dense
	Output  mat.Dense
}

func (activation *Activation_ReLU) Forward(inputs *mat.Dense) {
	activation.inputs = *mat.DenseCopyOf(inputs)

	activation.Output = *mat.DenseCopyOf(inputs)
	activation.Output.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return v
		} else {
			return 0
		}
	}, &activation.Output)
}

func (activation *Activation_ReLU) Backward(dvalues *mat.Dense) {
	activation.DInputs = *mat.DenseCopyOf(dvalues)
	activation.DInputs.Apply(func(i, j int, v float64) float64 {
		if v <= 0 {
			return 0
		}
		return v
	}, &activation.DInputs)
}
