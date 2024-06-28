package loss

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type MeanSquaredErrorLoss struct {
	DInputs mat.Dense
}

func (loss *MeanSquaredErrorLoss) Name() string {
	return "Mean Squared Error Loss"
}

func (loss *MeanSquaredErrorLoss) Calculate(prediction *mat.Dense, target []float64) float64 {
	sampleLosses := loss.Forward(prediction, target)
	value := floats.Sum(sampleLosses) / float64(len(sampleLosses))
	return value
}

// predictions has one value for the sample
// this is why target is single-dim array
func (loss *MeanSquaredErrorLoss) Forward(prediction *mat.Dense, target []float64) []float64 {
	result := make([]float64, len(target))

	data := prediction.RawMatrix().Data

	for i, item := range data {
		result[i] = math.Pow(target[i]-item, 2)
	}

	return result
}

func (loss *MeanSquaredErrorLoss) Backward(dvalues *mat.Dense, target []float64) {
	sampleCount, outputCount := dvalues.Dims()

	loss.DInputs = *mat.DenseCopyOf(dvalues)
	loss.DInputs.Apply(func(i, j int, v float64) float64 {
		return -2.0 * (target[i] - dvalues.At(i, j)) / float64(outputCount) / float64(sampleCount)
	}, &loss.DInputs)
}
