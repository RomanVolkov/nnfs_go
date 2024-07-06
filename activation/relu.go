package activation

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Activation_ReLU struct {
	inputs  mat.Dense
	DInputs mat.Dense
	Output  mat.Dense
}

func (a *Activation_ReLU) Name() string {
	return "RELU Activation"
}

func (activation *Activation_ReLU) Forward(inputs *mat.Dense, isTraining bool) {
	activation.inputs = *mat.DenseCopyOf(inputs)

	activation.Output = *mat.DenseCopyOf(inputs)
	activation.Output.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return v
		} else {
			return 0
		}
	}, &activation.Output)
}

func (activation *Activation_ReLU) Backward(dvalues *mat.Dense) {
	activation.DInputs = *mat.DenseCopyOf(dvalues)
	activation.DInputs.Apply(func(i, j int, v float64) float64 {
		if activation.inputs.At(i, j) <= 0 {
			return 0
		}
		return v
	}, &activation.DInputs)
}

func (a *Activation_ReLU) GetOutput() *mat.Dense {
	return &a.Output
}

func (a *Activation_ReLU) GetDInputs() *mat.Dense {
	return &a.DInputs
}

func (a *Activation_ReLU) Predictions(outputs *mat.Dense) mat.Dense {
	r, _ := outputs.Dims()
	prediction := make([]float64, r)

	for i := 0; i < r; i++ {
		rowValues := outputs.RawRowView(i)
		maxIndex := floats.MaxIdx(rowValues)
		prediction[i] = float64(maxIndex)
	}
	return *mat.NewDense(r, 1, prediction)
}
