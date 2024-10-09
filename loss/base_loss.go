package loss

import (
	"main/layer"
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type BaseLoss struct {
	DInputs            mat.Dense
	layers             []*layer.DenseLayer
	accumulatedLossSum float64
	accumulatedCount   int64
}

func (loss *BaseLoss) SetLayers(layers []*layer.DenseLayer) {
	loss.layers = layers
}

func CalculateLoss(loss LossInterface, prediction *mat.Dense, target *mat.Dense) float64 {
	sampleLosses := loss.Forward(prediction, target)
	value := floats.Sum(sampleLosses) / float64(len(sampleLosses))

	accumulatedLossSum := floats.Sum(sampleLosses)
	accumulatedCount := len(sampleLosses)
	loss.AddAccumulated(accumulatedLossSum, int64(accumulatedCount))

	return value
}

func (loss *BaseLoss) AddAccumulated(lossSumm float64, count int64) {
	loss.accumulatedCount += count
	loss.accumulatedLossSum += lossSumm
}

func (loss *BaseLoss) CalculateAccumulatedLoss() float64 {
	return loss.accumulatedLossSum / float64(loss.accumulatedCount)
}

func (loss *BaseLoss) ResetAccumulated() {
	loss.accumulatedCount = 0
	loss.accumulatedLossSum = 0.0
}

func (loss *BaseLoss) GetAccumulatedLossSum() float64 {
	return loss.accumulatedLossSum
}

func (loss *BaseLoss) GetAccumulatedCount() int64 {
	return loss.accumulatedCount
}

func (loss *BaseLoss) RegularizationLoss() float64 {
	value := 0.0

	for _, layer := range loss.layers {
		if layer.L1.Weight > 0 {
			temp := 0.0
			for i := range layer.Weights.RawMatrix().Data {
				temp += math.Abs(layer.Weights.RawMatrix().Data[i])
			}
			value += layer.L1.Weight * temp
		}
		if layer.L1.Bias > 0 {
			temp := 0.0
			for i := range layer.Biases.RawMatrix().Data {
				temp += math.Abs(layer.Biases.RawMatrix().Data[i])
			}
			value += layer.L1.Bias * temp
		}

		if layer.L2.Weight > 0 {
			temp := 0.0
			for i := range layer.Weights.RawMatrix().Data {
				temp += math.Pow(layer.Weights.RawMatrix().Data[i], 2)
			}
			value += layer.L2.Weight * temp
		}
		if layer.L2.Bias > 0 {
			temp := 0.0
			for i := range layer.Biases.RawMatrix().Data {
				temp += math.Pow(layer.Biases.RawMatrix().Data[i], 2)
			}
			value += layer.L2.Bias * temp
		}
	}
	return value
}
