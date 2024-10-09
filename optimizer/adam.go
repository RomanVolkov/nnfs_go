package optimizer

import (
	"main/layer"
	"math"

	"gonum.org/v1/gonum/mat"
)

type OptimizerAdam struct {
	CurrentLearningRate float64 `json:"currentLearningRate"`
	LearningRate        float64 `json:"learningRate"`
	Decay               float64 `json:"decay"`
	Epsilon             float64 `json:"epsilon"`
	Beta1               float64 `json:"beta1"`
	Beta2               float64 `json:"beta2"`
	Iterations          int     `json:"iterations"`
}

func NewAdam() OptimizerAdam {
	return OptimizerAdam{
		CurrentLearningRate: 0.001,
		LearningRate:        0.001,
		Decay:               0.,
		Epsilon:             1e-7,
		Beta1:               0.9,
		Beta2:               0.999,
		Iterations:          0,
	}
}

func (a *OptimizerAdam) Name() string {
	return "Adam Optimizer"
}

func (a *OptimizerAdam) GetCurrentLearningRate() float64 {
	return a.CurrentLearningRate
}

func (optimizer *OptimizerAdam) PreUpdate() {
	if optimizer.Decay > 0.0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1.0 / (1.0 + optimizer.Decay*float64(optimizer.Iterations)))
	}
}

func (optimizer *OptimizerAdam) UpdateParams(layer *layer.DenseLayer) {
	weightUpdates := mat.DenseCopyOf(&layer.Weights)
	biasUpdates := mat.DenseCopyOf(&layer.Biases)

	if layer.WeightCache == nil {
		layer.WeightCache = mat.DenseCopyOf(&layer.Weights)
		layer.WeightCache.Zero()
		layer.BiasCache = mat.DenseCopyOf(&layer.Biases)
		layer.BiasCache.Zero()
		layer.WeightMomentums = mat.DenseCopyOf(&layer.Weights)
		layer.WeightMomentums.Zero()
		layer.BiasMomentums = mat.DenseCopyOf(&layer.Biases)
		layer.BiasMomentums.Zero()
	}

	// momentum
	layer.WeightMomentums.Apply(func(i, j int, v float64) float64 {
		return optimizer.Beta1*v + (1-optimizer.Beta1)*layer.DWeights.At(i, j)
	}, layer.WeightMomentums)
	layer.BiasMomentums.Apply(func(i, j int, v float64) float64 {
		return optimizer.Beta1*v + (1-optimizer.Beta1)*layer.DBiases.At(i, j)
	}, layer.BiasMomentums)

	// get corrected momentum
	weightMomentumsCorrected := mat.DenseCopyOf(layer.WeightMomentums)
	biasMomentumsCorrected := mat.DenseCopyOf(layer.BiasMomentums)
	weightMomentumsCorrected.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(optimizer.Beta1, float64(optimizer.Iterations)+1))
	}, weightMomentumsCorrected)
	biasMomentumsCorrected.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(optimizer.Beta1, float64(optimizer.Iterations)+1))
	}, biasMomentumsCorrected)

	//  Update cache with squared current gradients

	layer.WeightCache.Apply(func(i, j int, v float64) float64 {
		return optimizer.Beta2*layer.WeightCache.At(i, j) + (1.-optimizer.Beta2)*math.Pow(layer.DWeights.At(i, j), 2)
	}, layer.WeightCache)
	layer.BiasCache.Apply(func(i, j int, v float64) float64 {
		return optimizer.Beta2*layer.BiasCache.At(i, j) + (1.-optimizer.Beta2)*math.Pow(layer.DBiases.At(i, j), 2)
	}, layer.BiasCache)

	// get corrected cache
	weightCacheCorrected := mat.DenseCopyOf(layer.WeightCache)
	biasCacheCorrected := mat.DenseCopyOf(layer.BiasCache)
	weightCacheCorrected.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(optimizer.Beta2, float64(optimizer.Iterations)+1))
	}, weightCacheCorrected)
	biasCacheCorrected.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(optimizer.Beta2, float64(optimizer.Iterations)+1))
	}, biasCacheCorrected)

	// Vanilla SGD + normalization with square rooted cache
	weightUpdates.Apply(func(i, j int, v float64) float64 {
		return -1 * optimizer.CurrentLearningRate * weightMomentumsCorrected.At(i, j) / (math.Sqrt(weightCacheCorrected.At(i, j)) + optimizer.Epsilon)
	}, weightUpdates)
	biasUpdates.Apply(func(i, j int, v float64) float64 {
		return -1 * optimizer.CurrentLearningRate * biasMomentumsCorrected.At(i, j) / (math.Sqrt(biasCacheCorrected.At(i, j)) + optimizer.Epsilon)
	}, biasUpdates)

	// update weights an biases
	layer.Weights.Add(&layer.Weights, weightUpdates)
	layer.Biases.Add(&layer.Biases, biasUpdates)
}

func (optimizer *OptimizerAdam) PostUpdate() {
	optimizer.Iterations += 1
}
