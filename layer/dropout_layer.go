package layer

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type DropoutLayer struct {
	inputs     mat.Dense
	binaryMask mat.Dense

	Output  mat.Dense
	DInputs mat.Dense

	// dropout rate
	Rate float64
}

func (layer *DropoutLayer) Name() string {
	return "Dropout Layer"
}

func (layer *DropoutLayer) Initialization(rate float64) *DropoutLayer {
	layer.Rate = 1.0 - rate
	return layer
}

func (layer *DropoutLayer) Forward(inputs *mat.Dense, isTraining bool) {
	layer.inputs = *inputs

	// if not in the training mode
	if !isTraining {
		layer.Output = *mat.DenseCopyOf(inputs)
		return
	}

	layer.binaryMask = *mat.DenseCopyOf(inputs)
	src := rand.New(rand.NewSource(1))
	layer.binaryMask.Apply(func(i, j int, v float64) float64 {
		return distuv.Binomial{N: 1, P: layer.Rate, Src: src}.Rand() / layer.Rate
	}, &layer.binaryMask)

	layer.Output = *mat.DenseCopyOf(inputs)
	layer.Output.MulElem(inputs, &layer.binaryMask)
}

func (layer *DropoutLayer) Backward(dvalues *mat.Dense) {
	layer.DInputs = *mat.DenseCopyOf(dvalues)
	layer.DInputs.MulElem(dvalues, &layer.binaryMask)
}

func (a *DropoutLayer) GetOutput() *mat.Dense {
	return &a.Output
}

func (a *DropoutLayer) GetDInputs() *mat.Dense {
	return &a.DInputs
}
