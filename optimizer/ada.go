package optimizer

import (
	"main/layer"
	"math"

	"gonum.org/v1/gonum/mat"
)

type OptimizerAda struct {
	CurrentLearningRate float64
	LearningRate        float64
	Decay               float64
	Epsilon             float64
	iterations          int
}

func NewAda(learningRate float64, decay float64, epsilon float64) OptimizerAda {
	return OptimizerAda{
		CurrentLearningRate: learningRate,
		LearningRate:        learningRate,
		Decay:               decay,
		Epsilon:             epsilon,
		iterations:          0,
	}
}

func (a *OptimizerAda) Name() string {
	return "Ada Optimizer"
}

func (optimizer *OptimizerAda) PreUpdate() {
	if optimizer.Decay > 0.0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1.0 / (1.0 + optimizer.Decay*float64(optimizer.iterations)))
	}
}

func (optimizer *OptimizerAda) UpdateParams(layer *layer.Layer) {
	weightUpdates := mat.DenseCopyOf(&layer.Weights)
	biasUpdates := mat.DenseCopyOf(&layer.Biases)

	if layer.WeightCache == nil {
		layer.WeightCache = mat.DenseCopyOf(&layer.Weights)
		layer.WeightCache.Zero()
		layer.BiasCache = mat.DenseCopyOf(&layer.Biases)
		layer.BiasCache.Zero()
	}

	layer.WeightCache.Apply(func(i, j int, v float64) float64 {
		return v + math.Pow(layer.DWeights.At(i, j), 2)
	}, layer.WeightCache)
	layer.BiasCache.Apply(func(i, j int, v float64) float64 {
		return v + math.Pow(layer.DBiases.At(i, j), 2)
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

func (optimizer *OptimizerAda) PostUpdate() {
	optimizer.iterations += 1
}
