package activation

import (
	"main/layer"

	"gonum.org/v1/gonum/mat"
)

type ActivationInterface interface {
	layer.LayerInterface
	Predictions(outputs *mat.Dense) mat.Dense
}
