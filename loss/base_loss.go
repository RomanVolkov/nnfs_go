package loss

import (
	"main/layer"
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type BaseLoss struct {
	DInputs mat.Dense
	layers  []*layer.Layer
}

func (loss *BaseLoss) SetLayers(layers []*layer.Layer) {
	loss.layers = layers
}

func CalculateLoss(loss LossInterface, prediction *mat.Dense, target *mat.Dense) float64 {
	sampleLosses := loss.Forward(prediction, target)
	value := floats.Sum(sampleLosses) / float64(len(sampleLosses))
	return value
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
