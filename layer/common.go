package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type LayerInterface interface {
	GetOutput() *mat.Dense
	GetDInputs() *mat.Dense
	Name() string
	Forward(inputs *mat.Dense, isTraining bool)
	// ??
	// Forward (layer *LayerInterface, isTraining bool)
	Backward(dvalues *mat.Dense)
}

// takes raw data from one sample for inputs and slices it according to InputShape
// e.g., Grayscake will return one mat.Dense
// RGB - len == 3
func ConvertSampleData(inputRawData []float64, shape InputShape) []mat.Dense {
	data := make([]mat.Dense, shape.Depths)
	for k := 0; k < shape.Depths; k++ {
		slice := inputRawData[k*shape.Height*shape.Width : (k+1)*shape.Height*shape.Width]
		data[k] = *mat.NewDense(shape.Height, shape.Width, slice)
	}

	return data
}

func MaxValue(m mat.Matrix) float64 {
	r, c := m.Dims()
	maxValue := float64(math.MinInt64)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			maxValue = max(maxValue, m.At(i, j))
		}
	}

	return maxValue
}
