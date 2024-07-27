package layer

import "gonum.org/v1/gonum/mat"

type LayerInterface interface {
	GetOutput() *mat.Dense
	GetDInputs() *mat.Dense
	Name() string
	Forward(inputs *mat.Dense, isTraining bool)
	Backward(dvalues *mat.Dense)
	// json.Marshaler
	// json.Unmarshaler
}
