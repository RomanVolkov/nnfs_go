package loss

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type BinaryCrossentropyLoss struct {
	BaseLoss
}

func (loss *BinaryCrossentropyLoss) Name() string {
	return "Binary Crossentropy Loss"
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

	tmp := mat.DenseCopyOf(predictionClipped)
	tmp.Apply(func(i, j int, v float64) float64 {
		return -1.0 * (float64(target[i])*math.Log(v) + (1.0-float64(target[i]))*math.Log(1.0-v))
	}, tmp)

	sampleLosses := make([]float64, len(target))
	r, c := tmp.Dims()
	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			sum += tmp.At(i, j)
		}
		sampleLosses[i] = sum / float64(c)
	}

	return sampleLosses
}

func (loss *BinaryCrossentropyLoss) Backward(dvalues *mat.Dense, target []uint8) {
	dvaluesClipped := mat.DenseCopyOf(dvalues)
	dvaluesClipped.Apply(func(i, j int, v float64) float64 {
		minValue := 1e-7
		maxValue := 1 - 1e-7
		return math.Max(math.Min(v, maxValue), minValue)
	}, dvalues)

	sampleCount, outputsCount := dvalues.Dims()

	loss.DInputs = *mat.NewDense(sampleCount, outputsCount, nil)
	for i := 0; i < sampleCount; i++ {
		for j := 0; j < outputsCount; j++ {
			value := -(float64(target[i])/dvalues.At(i, j) - float64(1-target[i])/(1.0-dvalues.At(i, j))) / float64(outputsCount) / float64(sampleCount)
			loss.DInputs.Set(i, j, value)
		}
	}
}
