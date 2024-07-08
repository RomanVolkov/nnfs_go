package loss

import (
	"main/utils"
	"math"

	"gonum.org/v1/gonum/mat"
)

type MeanSquaredErrorLoss struct {
	BaseLoss
}

func (loss *MeanSquaredErrorLoss) Name() string {
	return "Mean Squared Error Loss"
}

// predictions has one value for the sample
// this is why target is single-dim array
func (loss *MeanSquaredErrorLoss) Forward(prediction *mat.Dense, target *mat.Dense) []float64 {
	if !utils.CompareDims(prediction, target) {
		panic("incorrect dimentions")
	}
	rows, cols := prediction.Dims()

	result := make([]float64, rows)
	for i := range result {
		sum := 0.
		for j := 0; j < cols; j++ {
			sum += math.Pow(prediction.At(i, j)-target.At(i, j), 2)
		}
		result[i] = sum / float64(cols)
	}

	return result
}

func (loss *MeanSquaredErrorLoss) Backward(dvalues *mat.Dense, target *mat.Dense) {
	sampleCount, outputCount := dvalues.Dims()

	loss.DInputs = *mat.DenseCopyOf(dvalues)
	loss.DInputs.Apply(func(i, j int, v float64) float64 {
		return -2.0 * (target.At(i, j) - dvalues.At(i, j)) / float64(outputCount) / float64(sampleCount)
	}, &loss.DInputs)
}

func (loss *MeanSquaredErrorLoss) GetDInputs() *mat.Dense {
	return &loss.DInputs
}
