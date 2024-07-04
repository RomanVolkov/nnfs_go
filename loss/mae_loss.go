package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type MeanAbsoluteErrorLoss struct {
	BaseLoss
}

func (loss *MeanAbsoluteErrorLoss) Name() string {
	return "Mean Absolute Error Loss"
}

// predictions has one value for the sample
// this is why target is single-dim array
func (loss *MeanAbsoluteErrorLoss) Forward(prediction *mat.Dense, target []float64) []float64 {
	result := make([]float64, len(target))

	data := prediction.RawMatrix().Data

	for i, item := range data {
		result[i] = math.Abs(target[i] - item)
	}

	return result
}

func (loss *MeanAbsoluteErrorLoss) Backward(dvalues *mat.Dense, target []float64) {
	sampleCount, outputCount := dvalues.Dims()

	loss.DInputs = *mat.DenseCopyOf(dvalues)
	loss.DInputs.Apply(func(i, j int, v float64) float64 {
		return sign(target[i]-dvalues.At(i, j)) / float64(outputCount) / float64(sampleCount)
	}, &loss.DInputs)
}

func sign(value float64) float64 {
	if value < 0 {
		return -1
	} else if value == 0 {
		return 0
	} else {
		return 1
	}
}
