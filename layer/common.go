package layer

import "gonum.org/v1/gonum/mat"

type LayerInterface interface {
	GetOutput() *mat.Dense
	GetDInputs() *mat.Dense
	Name() string
	Forward(inputs *mat.Dense, isTraining bool)
	// ??
	// Forward (layer *LayerInterface, isTraining bool)
	Backward(dvalues *mat.Dense)
}
