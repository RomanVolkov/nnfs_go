package ops

import (
	"errors"
	"main/utils"

	"gonum.org/v1/gonum/mat"
)

func Correlate2dOps(input mat.Matrix, kernel mat.Matrix) (float64, error) {
	if !utils.CompareDims(&input, &kernel) {
		return 0, errors.New("size mismatch between input slice and kernel")
	}

	sum := 0.
	r, c := input.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += input.At(i, j) * kernel.At(i, j)
		}
	}

	return sum, nil
}

// input - 2D array of image (currentl grayscale?)
// kernel - 2D array
func Correlate2dValid(input mat.Dense, kernel mat.Dense) (mat.Dense, error) {
	// allocate output based on formula M-N+1
	// calculate the output
	n, _ := input.Dims()
	m, _ := kernel.Dims()
	s := n - m + 1
	output := mat.NewDense(s, s, nil)

	for i := 0; i < s; i++ {
		for j := 0; j < s; j++ {
			value, err := Correlate2dOps(input.Slice(i, i+m, j, j+m), &kernel)
			if err != nil {
				return *mat.NewDense(1, 1, nil), err
			}
			output.Set(i, j, value)
		}
	}

	return *output, nil
}

func Correlate2DFull(input mat.Dense, kernel mat.Dense) (mat.Dense, error) {
	inputSize, _ := input.Dims()
	kernelSize, _ := kernel.Dims()
	s := inputSize + 2 - kernelSize + 1
	output := mat.NewDense(s, s, nil)

	expandedInput := mat.NewDense(inputSize+2, inputSize+2, nil)
	expandedInput.Zero()

	// copy data from input taking Padding into account
	for i := 0; i < inputSize; i++ {
		for j := 0; j < inputSize; j++ {
			expandedInput.Set(i+1, j+1, input.At(i, j))
		}
	}

	for i := 0; i < s; i++ {
		for j := 0; j < s; j++ {
			value, err := Correlate2dOps(expandedInput.Slice(i, i+kernelSize, j, j+kernelSize), &kernel)
			if err != nil {
				return *mat.NewDense(1, 1, nil), err
			}
			output.Set(i, j, value)
		}
	}

	return *output, nil
}
