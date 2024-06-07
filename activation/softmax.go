package activation

import (
	"math"
	"slices"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type SoftmaxActivation struct {
	Output  mat.Dense
	DInputs mat.Dense
}

func (activation *SoftmaxActivation) Forward(inputs *mat.Dense) {
	r, c := inputs.Dims()
	activation.Output = *mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		row := softmax(inputs.RawRowView(i))
		activation.Output.SetRow(i, row)
	}
}

func softmax(input []float64) []float64 {
	output := make([]float64, len(input))
	expValues := make([]float64, len(input))

	maxValue := slices.Max(input)
	for i := range expValues {
		expValues[i] = math.Exp(input[i] - maxValue)
	}

	sum := floats.Sum(expValues)

	for i := range output {
		output[i] = expValues[i] / sum
	}
	return output
}

func (activation *SoftmaxActivation) Backward(dvalues *mat.Dense) {
	activation.DInputs = *mat.DenseCopyOf(dvalues)
	r, _ := dvalues.Dims()
	// r is number of samples
	for index := 0; index < r; index++ {
		singleOutputData := activation.Output.RawRowView(index)
		col := len(singleOutputData)
		singleDvalues := dvalues.RowView(index)
		singleOutput := mat.NewDense(col, 1, singleOutputData)

		jacobianMatrix := mat.NewDense(col, col, nil)

		diagflagOutput := diagflat(singleOutputData)
		dotProductOutput := mat.NewDense(col, col, nil)
		dotProductOutput.Product(singleOutput, singleOutput.T())

		jacobianMatrix.Sub(&diagflagOutput, dotProductOutput)

		m, _ := jacobianMatrix.Dims()
		_, p := singleDvalues.Dims()

		output := mat.NewDense(m, p, nil)
		output.Product(jacobianMatrix, singleDvalues)

		activation.DInputs.SetRow(index, output.RawMatrix().Data)
	}
}

func diagflat(values []float64) mat.Dense {
	len := len(values)
	result := mat.NewDense(len, len, nil)
	for i := 0; i < len; i++ {
		for j := 0; j < len; j++ {
			if j == i {
				result.Set(i, j, values[i])
			} else {
				result.Set(i, j, 0)
			}
		}
	}
	return *result
}
