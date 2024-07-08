package loss

import (
	"main/utils"
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
func (loss *MeanAbsoluteErrorLoss) Forward(prediction *mat.Dense, target *mat.Dense) []float64 {
	if !utils.CompareDims(prediction, target) {
		panic("incorrect dimentions")
	}
	rows, cols := prediction.Dims()

	result := make([]float64, rows)
	for i := range result {
		sum := 0.
		for j := 0; j < cols; j++ {
			sum += math.Abs(prediction.At(i, j) - target.At(i, j))
		}
		result[i] = sum / float64(cols)
	}

	return result
}

func (loss *MeanAbsoluteErrorLoss) Backward(dvalues *mat.Dense, target *mat.Dense) {
	sampleCount, outputCount := dvalues.Dims()

	loss.DInputs = *mat.DenseCopyOf(dvalues)
	loss.DInputs.Apply(func(i, j int, v float64) float64 {
		return sign(target.At(i, j)-dvalues.At(i, j)) / float64(outputCount) / float64(sampleCount)
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
