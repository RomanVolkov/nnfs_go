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

func (layer *DropoutLayer) Initialization(rate float64) {
	layer.Rate = 1.0 - rate
}

func (layer *DropoutLayer) Forward(inputs *mat.Dense) {
	layer.inputs = *inputs

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
