package optimizer

import (
	"main/layer"
	"math"

	"gonum.org/v1/gonum/mat"
)

type OptimizerRMSprop struct {
	CurrentLearningRate float64
	LearningRate        float64
	Decay               float64
	Epsilon             float64
	Rho                 float64
	iterations          int
}

func NewRMSprop(learningRate float64, decay float64, epsilon float64, rho float64) OptimizerRMSprop {
	return OptimizerRMSprop{
		CurrentLearningRate: learningRate,
		LearningRate:        learningRate,
		Decay:               decay,
		Epsilon:             epsilon,
		Rho:                 rho,
		iterations:          0,
	}
}

func (optimizer *OptimizerRMSprop) PreUpdate() {
	if optimizer.Decay > 0.0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1.0 / (1.0 + optimizer.Decay*float64(optimizer.iterations)))
	}
}

func (optimizer *OptimizerRMSprop) UpdateParams(layer *layer.Layer) {
	weightUpdates := mat.DenseCopyOf(&layer.Weights)
	biasUpdates := mat.DenseCopyOf(&layer.Biases)

	if layer.WeightCache == nil {
		layer.WeightCache = mat.DenseCopyOf(&layer.Weights)
		layer.WeightCache.Zero()
		layer.BiasCache = mat.DenseCopyOf(&layer.Biases)
		layer.BiasCache.Zero()
	}

	layer.WeightCache.Apply(func(i, j int, v float64) float64 {
		return optimizer.Rho*layer.WeightCache.At(i, j) + (1.-optimizer.Rho)*math.Pow(layer.DWeights.At(i, j), 2)
	}, layer.WeightCache)
	layer.BiasCache.Apply(func(i, j int, v float64) float64 {
		return optimizer.Rho*layer.BiasCache.At(i, j) + (1.-optimizer.Rho)*math.Pow(layer.DBiases.At(i, j), 2)
	}, layer.BiasCache)

	weightUpdates.Apply(func(i, j int, v float64) float64 {
		return -1 * optimizer.CurrentLearningRate * layer.DWeights.At(i, j) / (math.Sqrt(layer.WeightCache.At(i, j)) + optimizer.Epsilon)
	}, weightUpdates)
	biasUpdates.Apply(func(i, j int, v float64) float64 {
		return -1 * optimizer.CurrentLearningRate * layer.DBiases.At(i, j) / (math.Sqrt(layer.BiasCache.At(i, j)) + optimizer.Epsilon)
	}, biasUpdates)

	// update weights an biases
	layer.Weights.Add(&layer.Weights, weightUpdates)
	layer.Biases.Add(&layer.Biases, biasUpdates)
}

func (optimizer *OptimizerRMSprop) PostUpdate() {
	optimizer.iterations += 1
}
