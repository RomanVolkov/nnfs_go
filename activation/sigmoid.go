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

func (a *SigmoidActivation) GetOutput() *mat.Dense {
	return &a.Output
}

func (a *SigmoidActivation) GetDInputs() *mat.Dense {
	return &a.DInputs
}

func (a *SigmoidActivation) Predictions(outputs *mat.Dense) mat.Dense {
	prediction := mat.DenseCopyOf(outputs)

	prediction.Apply(func(i, j int, v float64) float64 {
		if v > 0.5 {
			return 1.
		} else {
			return 0.
		}
	}, prediction)
	return *prediction
}
