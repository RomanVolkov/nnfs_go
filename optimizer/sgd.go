package optimizer

import (
	"main/layer"

	"gonum.org/v1/gonum/mat"
)

type OptimizerSGD struct {
	LearningRate float64
}

func NewSGD() OptimizerSGD {
	return OptimizerSGD{LearningRate: 1.0}
}

func (optimizer *OptimizerSGD) UpdateParams(layer *layer.Layer) {
	dweights := mat.DenseCopyOf(&layer.DWeights)
	dbiases := mat.DenseCopyOf(&layer.DBiases)

	dweights.Apply(func(i, j int, v float64) float64 {
		return v * (-1) * optimizer.LearningRate
	}, dweights)
	dbiases.Apply(func(i, j int, v float64) float64 {
		return v * (-1) * optimizer.LearningRate
	}, dbiases)

	layer.Weights.Add(&layer.Weights, dweights)
	layer.Biases.Add(&layer.Biases, dbiases)
}
