package dataset

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func VerticalData(samples int, classes int) (*mat.Dense, *mat.Dense) {
	x := mat.NewDense(samples*classes, 2, nil)
	y := mat.NewDense(samples*classes, 1, nil)

	for class_number := 0; class_number < classes; class_number++ {
		for ix := class_number * samples; ix < samples*(class_number+1); ix++ {

			x.Set(ix, 0, randomValue()*0.1+float64(class_number)/3.0)
			x.Set(ix, 1, randomValue()*0.1+0.5)
			y.Set(ix, 0, float64(class_number))
		}
	}
	return x, y
}

func randomValue() float64 {
	return float64(rand.Intn(100)) / 100.0
}
