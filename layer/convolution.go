package layer

import (
	"gonum.org/v1/gonum/mat"
)

type ConvolutionLayer struct{}

// inputShape - side of input image (width or heigh)
// filterSize - size of kernel's side
// filterNumber - number of kernels
func (layer *ConvolutionLayer) Initialization(inputSize int, kernelSize int, kernelCount int) (outputSize int) {
	panic(10)

}

func (layer *ConvolutionLayer) Forward(inputs *mat.Dense, isTraining bool) {

}

// dvalues comes form max pooling (well, via relu)
func (layer *ConvolutionLayer) Backward(dvalues *mat.Dense) {

}

func (layer *ConvolutionLayer) Name() string {
	return "ConvolutionLayer"
}

func (layer *ConvolutionLayer) GetOutput() *mat.Dense {
	panic(1)
}

func (layer *ConvolutionLayer) GetDInputs() *mat.Dense {
	panic(1)
}
