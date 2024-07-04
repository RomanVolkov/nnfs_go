package dataset

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func SineData(samples int) (mat.Dense, mat.Dense) {
	x, y := make([]float64, samples), make([]float64, samples)
	for i := 0; i < samples; i++ {
		x[i] = float64(i) / float64(samples)
		y[i] = math.Sin(2 * math.Pi * x[i])
	}

	// TODO: define dimentions for Y
	return *mat.NewDense(samples, 1, x), *mat.NewDense(samples, 1, y)
}
