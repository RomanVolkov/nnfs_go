package model

import (
	"main/layer"

	"gonum.org/v1/gonum/mat"
)

type LayerInterface interface {
	Name() string
	Forward(inputs *mat.Dense)
	Backward(dvalues *mat.Dense)
}

type LossInterface interface {
	Name() string
	Calculate(prediction *mat.Dense, target []float64) float64
	Forward(prediction *mat.Dense, target []float64) []float64
	Backward(dvalues *mat.Dense, target []float64)
}

type OptimizerInterface interface {
	Name() string
	PreUpdate()
	UpdateParams(layer *layer.Layer)
	PostUpdate()
}
