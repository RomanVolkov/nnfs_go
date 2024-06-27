package dataset

import "math"

func SineData(samples int) ([]float64, []float64) {
	x, y := make([]float64, samples), make([]float64, samples)
	for i := 0; i < samples; i++ {
		x[i] = float64(i) / float64(samples)
		y[i] = math.Sin(2 * math.Pi * x[i])
	}

	return x, y
}
