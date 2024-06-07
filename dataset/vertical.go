package dataset

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func VerticalData(samples int, classes int) (*mat.Dense, []uint8) {
	x := mat.NewDense(samples*classes, 2, nil)
	y := make([]uint8, samples*classes)

	for class_number := 0; class_number < classes; class_number++ {
		for ix := class_number * samples; ix < samples*(class_number+1); ix++ {

			x.Set(ix, 0, randomValue()*0.1+float64(class_number)/3.0)
			x.Set(ix, 1, randomValue()*0.1+0.5)
			y[ix] = uint8(class_number)
		}
	}
	return x, y
}

func randomValue() float64 {
	return float64(rand.Intn(100)) / 100.0
}
