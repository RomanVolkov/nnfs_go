package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SigmoidActivation struct {
	inputs mat.Dense

	Output  mat.Dense
	DInputs mat.Dense
}

func (activation *SigmoidActivation) Forward(inputs mat.Dense) {
	activation.inputs = inputs
	activation.Output = *mat.DenseCopyOf(&inputs)
	activation.Output.Apply(func(i, j int, v float64) float64 {
		return 1. / (1. + math.Exp(-1*v))
	}, &activation.Output)
}

func (activation *SigmoidActivation) Backward(dvalues mat.Dense) {
	activation.DInputs = *mat.DenseCopyOf(&dvalues)
	activation.DInputs.Apply(func(i, j int, v float64) float64 {
		return v * (1. - activation.Output.At(i, j)*activation.Output.At(i, j))
	}, &dvalues)
}
