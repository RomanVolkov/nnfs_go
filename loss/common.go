package loss

import (
	"main/layer"

	"gonum.org/v1/gonum/mat"
)

type LossInterface interface {
	Name() string
	GetDInputs() *mat.Dense
	SetLayers(layers []*layer.Layer)
	Forward(prediction *mat.Dense, target *mat.Dense) []float64
	Backward(dvalues *mat.Dense, target *mat.Dense)
	RegularizationLoss() float64
	AddAccumulated(lossSumm float64, count int64)
	ResetAccumulated()
}
