package loss

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type BinaryCrossentropyLoss struct {
	DInputs mat.Dense
}

func (loss *BinaryCrossentropyLoss) Calculate(prediction *mat.Dense, y []uint8) float64 {
	sampleLosses := loss.Forward(prediction, y)
	value := floats.Sum(sampleLosses) / float64(len(sampleLosses))
	return value
}

func (loss *BinaryCrossentropyLoss) Forward(prediction *mat.Dense, target []uint8) []float64 {
	predictionClipped := mat.DenseCopyOf(prediction)
	predictionClipped.Apply(func(i, j int, v float64) float64 {
		minValue := 1e-7
		maxValue := 1 - 1e-7
		return math.Max(math.Min(v, maxValue), minValue)
	}, prediction)

	return make([]float64, 0)
}

func (loss *BinaryCrossentropyLoss) Backward(dvalues mat.Dense, target []uint8) {
}
