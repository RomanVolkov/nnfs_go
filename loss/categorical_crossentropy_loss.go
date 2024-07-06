package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type CategoricalCrossentropyLoss struct {
	BaseLoss
}

func (loss *CategoricalCrossentropyLoss) Name() string {
	return "Categorial Crossentropy Loss"
}

func (loss *CategoricalCrossentropyLoss) Forward(prediction *mat.Dense, target []uint8) []float64 {
	predictionClipped := mat.DenseCopyOf(prediction)
	predictionClipped.Apply(func(i, j int, v float64) float64 {
		minValue := 1e-7
		maxValue := 1 - 1e-7
		return math.Max(math.Min(v, maxValue), minValue)
	}, prediction)

	confidences := make([]float64, len(target))
	for i := range target {
		idx := target[i]
		confidences[i] = predictionClipped.At(i, int(idx))
	}

	negative_log_likelihoods := make([]float64, len(confidences))
	for i := range confidences {
		negative_log_likelihoods[i] = -1 * math.Log(confidences[i])
	}

	return negative_log_likelihoods
}

func (loss *CategoricalCrossentropyLoss) Backward(dvalues mat.Dense, target []uint8) {
	samplesCounts, labelsCount := dvalues.Dims()

	loss.DInputs = *mat.NewDense(samplesCounts, labelsCount, nil)
	for i := 0; i < samplesCounts; i++ {
		for j := 0; j < labelsCount; j++ {
			value := 0.0
			// one-hot vector check
			if j == int(target[i]) {
				value = 1
				// calculation of gradient
				value = -1.0 * float64(value) / dvalues.At(i, j)
				// normalize gradient
				value = value / float64(samplesCounts)
			}
			loss.DInputs.Set(i, j, value)
		}
	}
}
