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

func (loss *CategoricalCrossentropyLoss) Forward(prediction *mat.Dense, target *mat.Dense) []float64 {
	r, c := target.Dims()
	if c != 1 {
		panic("target does not consist from target class; should have one value from sample")
	}
	predictionClipped := mat.DenseCopyOf(prediction)
	predictionClipped.Apply(func(i, j int, v float64) float64 {
		minValue := 1e-7
		maxValue := 1 - 1e-7
		return math.Max(math.Min(v, maxValue), minValue)
	}, prediction)

	confidences := make([]float64, r)
	for i := range confidences {
		idx := target.At(i, 0)
		confidences[i] = predictionClipped.At(i, int(idx))
	}

	negative_log_likelihoods := make([]float64, len(confidences))
	for i := range confidences {
		negative_log_likelihoods[i] = -1 * math.Log(confidences[i])
	}

	return negative_log_likelihoods
}

func (loss *CategoricalCrossentropyLoss) Backward(dvalues mat.Dense, target *mat.Dense) {
	_, c := target.Dims()
	if c != 1 {
		panic("target does not consist from target class; should have one value from sample")
	}
	samplesCounts, labelsCount := dvalues.Dims()

	loss.DInputs = *mat.NewDense(samplesCounts, labelsCount, nil)
	for i := 0; i < samplesCounts; i++ {
		for j := 0; j < labelsCount; j++ {
			value := 0.0
			// one-hot vector check
			if j == int(target.At(i, 0)) {
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
