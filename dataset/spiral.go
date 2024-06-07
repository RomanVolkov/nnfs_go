package dataset

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func SpiralData(samples int, classes int) (*mat.Dense, []uint8) {
	x := mat.NewDense(samples*classes, 2, nil)
	y := make([]uint8, samples*classes)

	for class_number := 0; class_number < classes; class_number++ {
		r := make([]float64, samples)
		for i := range r {
			r[i] = float64(i) / float64(samples-1)
		}

		t := make([]float64, samples)
		for i := range t {
			t[i] = float64(class_number*4) + float64(i)*4/float64(samples) + rand.NormFloat64()*0.2
		}
		for ix := class_number * samples; ix < samples*(class_number+1); ix++ {
			idx := ix - class_number*samples
			x.Set(ix, 0, r[idx]*math.Sin(t[idx]*2.5))
			x.Set(ix, 1, r[idx]*math.Cos(t[idx]*2.5))
			y[ix] = uint8(class_number)
		}
	}
	return x, y
}
