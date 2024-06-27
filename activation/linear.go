package activation

import "gonum.org/v1/gonum/mat"

type LinearActivation struct {
	inputs  mat.Dense
	Outputs mat.Dense
	DInputs mat.Dense
}

func (a *LinearActivation) Forward(inputs *mat.Dense) {
	a.inputs = *inputs
	a.Outputs = *inputs
}

func (a *LinearActivation) Backward(dvalues *mat.Dense) {
	a.DInputs = *mat.DenseCopyOf(dvalues)
}
