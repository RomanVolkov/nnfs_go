package layer

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	inputs mat.Dense

	DWeights mat.Dense
	DBiases  mat.Dense
	DInputs  mat.Dense

	// rows - number of inputs from previous Layer
	// cols - number of neurons within the Layer
	// each column corresponds to weights of a neuron
	Weights mat.Dense
	// one row with cols equal to number of neurons
	Biases mat.Dense

	// rows - number of inputs from previews Layer aka number of samples
	// cols - number of neurons from current Layer
	Output mat.Dense
}

func (layer *Layer) Initialization(n_inputs int, n_neurons int) {
	weights := make([]float64, n_inputs*n_neurons)
	for i := range weights {
		weights[i] = 0.01 * (rand.Float64() - 0.5) * 2
	}
	layer.Weights = *mat.NewDense(n_inputs, n_neurons, weights)
	biases := make([]float64, n_neurons)
	layer.Biases = *mat.NewDense(1, n_neurons, biases)
	layer.Biases.Zero()
}

func (layer *Layer) Forward(inputs *mat.Dense) {
	layer.inputs = *mat.DenseCopyOf(inputs)
	// number_rows is equal to input sample size
	number_rows, _ := inputs.Dims()
	_, number_cols := layer.Weights.Dims()

	result := mat.NewDense(number_rows, number_cols, nil)
	result.Product(inputs, &layer.Weights)

	rows, cols := result.Dims()
	// i - sample number
	for i := 0; i < rows; i++ {
		// add biases
		for j := 0; j < cols; j++ {
			value := result.At(i, j) + layer.Biases.At(0, j)
			result.Set(i, j, value)
		}
	}
	// having cols with neuron value
	layer.Output = *result
}

func (layer *Layer) Backward(dvalues *mat.Dense) {
	// Gradient on inputs
	m, _ := dvalues.Dims()
	_, p := layer.Weights.T().Dims()
	layer.DInputs = *mat.NewDense(m, p, nil)
	layer.DInputs.Product(dvalues, layer.Weights.T())

	// Gradients on params
	_, biasesCols := dvalues.Dims()
	layer.DBiases = *mat.NewDense(1, biasesCols, nil)
	for i := 0; i < biasesCols; i++ {
		layer.DBiases.Set(0, i, mat.Sum(dvalues.ColView(i)))
	}

	m, _ = layer.inputs.T().Dims()
	_, p = dvalues.Dims()
	layer.DWeights = *mat.NewDense(m, p, nil)
	layer.DWeights.Product(layer.inputs.T(), dvalues)
}
