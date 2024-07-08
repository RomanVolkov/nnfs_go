package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type BinaryCrossentropyLoss struct {
	BaseLoss
}

func (loss *BinaryCrossentropyLoss) Name() string {
	return "Binary Crossentropy Loss"
}

func (loss *BinaryCrossentropyLoss) Forward(prediction *mat.Dense, target *mat.Dense) []float64 {
	rows, cols := prediction.Dims()
	predictionClipped := mat.DenseCopyOf(prediction)
	predictionClipped.Apply(func(i, j int, v float64) float64 {
		minValue := 1e-7
		maxValue := 1 - 1e-7
		return math.Max(math.Min(v, maxValue), minValue)
	}, prediction)

	tmp := mat.DenseCopyOf(predictionClipped)
	tmp.Apply(func(i, j int, v float64) float64 {
		return -1.0 * (target.At(i, j)*math.Log(v) + (1.0-target.At(i, j))*math.Log(1.0-v))
	}, tmp)

	sampleLosses := make([]float64, rows)
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += tmp.At(i, j)
		}
		sampleLosses[i] = sum / float64(cols)
	}

	return sampleLosses
}

func (loss *BinaryCrossentropyLoss) Backward(dvalues *mat.Dense, target *mat.Dense) {
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
			value := -(target.At(i, j)/dvaluesClipped.At(i, j) - (1.0-target.At(i, j))/(1.0-dvaluesClipped.At(i, j)))
			value /= float64(outputsCount)
			value /= float64(sampleCount)
			loss.DInputs.Set(i, j, value)
		}
	}
}

func (loss *BinaryCrossentropyLoss) GetDInputs() *mat.Dense {
	return &loss.DInputs
}
