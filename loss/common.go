package loss

import (
	"main/layer"

	"gonum.org/v1/gonum/mat"
)

type LossInterface interface {
	Name() string
	SetLayers(layers []*layer.Layer)
	Calculate(prediction *mat.Dense, target []float64) float64
	Forward(prediction *mat.Dense, target []float64) []float64
	Backward(dvalues *mat.Dense, target []float64)
}
