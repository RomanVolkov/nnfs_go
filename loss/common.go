package loss

import (
	"main/layer"

	"gonum.org/v1/gonum/mat"
)

type LossInterface interface {
	Name() string
	GetDInputs() *mat.Dense
	SetLayers(layers []*layer.Layer)
	Forward(prediction *mat.Dense, target []float64) []float64
	Backward(dvalues *mat.Dense, target []float64)
	RegularizationLoss() float64
}
