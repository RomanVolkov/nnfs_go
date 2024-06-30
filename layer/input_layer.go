package layer

import (
	"gonum.org/v1/gonum/mat"
)

type InputLayer struct {
	Output  mat.Dense
	DInputs mat.Dense
}

func (layer *InputLayer) Name() string {
	return "Input Layer"
}

func (layer *InputLayer) Forward(inputs *mat.Dense) {
	layer.Output = *inputs
}

func (layer *InputLayer) Backward(dvalues *mat.Dense) {
	panic("should not be used")
}

func (a *InputLayer) GetOutput() *mat.Dense {
	return &a.Output
}

func (a *InputLayer) GetDInputs() *mat.Dense {
	panic("should not be used")
}
