package layer

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Regularizer struct {
	Weight, Bias float64
}

type Layer struct {
	inputs mat.Dense

	// rows - number of inputs from previous Layer
	// cols - number of neurons within the Layer
	// each column corresponds to weights of a neuron
	Weights mat.Dense
	// one row with cols equal to number of neurons
	Biases mat.Dense

	// rows - number of inputs from previews Layer aka number of samples
	// cols - number of neurons from current Layer
	Output mat.Dense

	// Backward pass
	DWeights mat.Dense
	DBiases  mat.Dense
	DInputs  mat.Dense

	// regularization strenght
	L1 Regularizer
	L2 Regularizer

	// used by SGD and Adam optimizers
	WeightMomentums *mat.Dense
	BiasMomentums   *mat.Dense

	// used by Ada, Adam and RMSProp optimizers
	WeightCache *mat.Dense
	BiasCache   *mat.Dense
}

func (layer *Layer) Name() string {
	return "Dense Layer"
}

func (layer *Layer) Initialization(n_inputs int, n_neurons int) *Layer {
	weights := make([]float64, n_inputs*n_neurons)
	for i := range weights {
		weights[i] = 0.01 * (rand.Float64() - 0.5) * 2
	}
	layer.Weights = *mat.NewDense(n_inputs, n_neurons, weights)
	biases := make([]float64, n_neurons)
	layer.Biases = *mat.NewDense(1, n_neurons, biases)
	layer.Biases.Zero()

	layer.L1 = Regularizer{0, 0}
	layer.L2 = Regularizer{0, 0}
	return layer
}

func (layer *Layer) Forward(inputs *mat.Dense, isTraining bool) {
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
	// Gradients on params
	_, biasesCols := dvalues.Dims()
	layer.DBiases = *mat.NewDense(1, biasesCols, nil)
	for i := 0; i < biasesCols; i++ {
		layer.DBiases.Set(0, i, mat.Sum(dvalues.ColView(i)))
	}

	m, _ := layer.inputs.T().Dims()
	_, p := dvalues.Dims()
	layer.DWeights = *mat.NewDense(m, p, nil)
	layer.DWeights.Product(layer.inputs.T(), dvalues)

	// Gradients on regularization
	// L1 on weights
	if layer.L1.Weight > 0 {
		dl1 := mat.DenseCopyOf(&layer.Weights)
		dl1.Apply(func(i, j int, v float64) float64 {
			if v >= 0.0 {
				return layer.L1.Weight
			} else {
				return -1 * layer.L1.Weight
			}
		}, dl1)
		layer.DWeights.Add(&layer.DWeights, dl1)
	}
	// L1 on biases
	if layer.L1.Bias > 0 {
		dl1 := mat.DenseCopyOf(&layer.Biases)
		dl1.Apply(func(i, j int, v float64) float64 {
			if v >= 0.0 {
				return layer.L1.Bias
			} else {
				return -1 * layer.L1.Bias
			}
		}, dl1)
		layer.DBiases.Add(&layer.DBiases, dl1)
	}
	// L2 on weights
	if layer.L2.Weight > 0 {
		tmp := mat.DenseCopyOf(&layer.Weights)
		tmp.Apply(func(i, j int, v float64) float64 {
			return 2 * layer.L2.Weight * v
		}, &layer.Weights)
		layer.DWeights.Add(&layer.DWeights, tmp)
	}
	// L2 on biases
	if layer.L2.Bias > 0 {
		tmp := mat.DenseCopyOf(&layer.Biases)
		tmp.Apply(func(i, j int, v float64) float64 {
			return 2 * layer.L2.Bias * v
		}, &layer.Biases)
		layer.DBiases.Add(&layer.DBiases, tmp)
	}

	// Gradient on inputs
	m, _ = dvalues.Dims()
	_, p = layer.Weights.T().Dims()
	layer.DInputs = *mat.NewDense(m, p, nil)
	layer.DInputs.Product(dvalues, layer.Weights.T())
}

func (a *Layer) GetOutput() *mat.Dense {
	return &a.Output
}

func (a *Layer) GetDInputs() *mat.Dense {
	return &a.DInputs
}
