package activation

import (
	"gonum.org/v1/gonum/mat"
)

type LinearActivation struct {
	inputs  mat.Dense
	Output  mat.Dense
	DInputs mat.Dense
}

func (a *LinearActivation) Name() string {
	return "Linear Activation"
}

func (a *LinearActivation) Forward(inputs *mat.Dense) {
	a.inputs = *inputs
	a.Output = *inputs
}

func (a *LinearActivation) Backward(dvalues *mat.Dense) {
	a.DInputs = *mat.DenseCopyOf(dvalues)
}

func (a *LinearActivation) GetOutput() *mat.Dense {
	return &a.Output
}

func (a *LinearActivation) GetDInputs() *mat.Dense {
	return &a.DInputs
}

func (a *LinearActivation) Predictions(outputs *mat.Dense) mat.Dense {
	return *mat.DenseCopyOf(outputs)
}
