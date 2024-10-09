package optimizer

import (
	"main/layer"

	"gonum.org/v1/gonum/mat"
)

type OptimizerSGD struct {
	CurrentLearningRate float64
	LearningRate        float64
	Decay               float64
	Momentum            float64
	iterations          int
}

func NewSGD(learningRate float64, decay float64, momentum float64) OptimizerSGD {
	return OptimizerSGD{
		CurrentLearningRate: learningRate,
		LearningRate:        learningRate,
		Decay:               decay,
		Momentum:            momentum,
		iterations:          0,
	}
}

func (a *OptimizerSGD) Name() string {
	return "SGD Optimizer"
}

func (optimizer *OptimizerSGD) PreUpdate() {
	if optimizer.Decay > 0.0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1.0 / (1.0 + optimizer.Decay*float64(optimizer.iterations)))
	}
}

func (optimizer *OptimizerSGD) UpdateParams(layer *layer.DenseLayer) {
	weightUpdates := mat.DenseCopyOf(&layer.Weights)
	biasUpdates := mat.DenseCopyOf(&layer.Biases)

	if optimizer.Momentum > 0.0 {
		if layer.WeightMomentums == nil {
			layer.WeightMomentums = mat.DenseCopyOf(&layer.Weights)
			layer.WeightMomentums.Zero()
			layer.BiasMomentums = mat.DenseCopyOf(&layer.Biases)
			layer.BiasMomentums.Zero()
		}
		// weights
		weightUpdates.Apply(func(i, j int, v float64) float64 {
			return optimizer.Momentum*layer.WeightMomentums.At(i, j) - optimizer.CurrentLearningRate*layer.DWeights.At(i, j)
		}, weightUpdates)
		layer.WeightMomentums = weightUpdates
		// bises
		biasUpdates.Apply(func(i, j int, v float64) float64 {
			return optimizer.Momentum*layer.BiasMomentums.At(i, j) - optimizer.CurrentLearningRate*layer.DBiases.At(i, j)
		}, biasUpdates)
		layer.BiasMomentums = biasUpdates
	} else {
		// vanilla SGD
		weightUpdates.Apply(func(i, j int, v float64) float64 {
			return (-1) * optimizer.CurrentLearningRate * layer.DWeights.At(i, j)
		}, weightUpdates)
		biasUpdates.Apply(func(i, j int, v float64) float64 {
			return (-1) * optimizer.CurrentLearningRate * layer.DBiases.At(i, j)
		}, biasUpdates)
	}

	// update weights an biases
	layer.Weights.Add(&layer.Weights, weightUpdates)
	layer.Biases.Add(&layer.Biases, biasUpdates)
}

func (optimizer *OptimizerSGD) PostUpdate() {
	optimizer.iterations += 1
}

func (optimizer *OptimizerSGD) GetCurrentLearningRate() float64 {
	return optimizer.CurrentLearningRate
}
