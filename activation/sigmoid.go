package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SigmoidActivation struct {
	Output  mat.Dense
	DInputs mat.Dense
}

func (a *SigmoidActivation) Name() string {
	return "Sigmoid Activation"
}

func (activation *SigmoidActivation) Forward(inputs *mat.Dense) {
	activation.Output = *mat.DenseCopyOf(inputs)
	activation.Output.Apply(func(i, j int, v float64) float64 {
		return 1. / (1. + math.Exp(-1*v))
	}, inputs)
}

func (activation *SigmoidActivation) Backward(dvalues *mat.Dense) {
	activation.DInputs = *mat.DenseCopyOf(dvalues)
	activation.DInputs.Apply(func(i, j int, v float64) float64 {
		return v * (1. - activation.Output.At(i, j)) * activation.Output.At(i, j)
	}, dvalues)
}
