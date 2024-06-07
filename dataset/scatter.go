package dataset

import (
	"fmt"
	"image/color"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func ScatterData(values *mat.Dense, classes_count int) {
	r, _ := values.Dims()
	x, y := make([]float64, r), make([]float64, r)
	for i := 0; i < r; i++ {
		x[i] = values.At(i, 0)
		y[i] = values.At(i, 1)
	}

	p := plot.New()

	p.Title.Text = "Data"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	// Draw a grid behind the data
	p.Add(plotter.NewGrid())

	for class := 0; class < classes_count; class++ {
		offset := r / classes_count
		fmt.Println(offset)
		pts := make(plotter.XYs, offset)
		for i := 0; i < offset; i++ {
			pts[i].X = x[class*offset+i]
			pts[i].Y = y[class*offset+i]
		}

		s, err := plotter.NewScatter(pts)
		s.Color = color.RGBA{R: uint8(rand.Intn(255)), B: uint8(rand.Intn(255)), A: uint8(rand.Intn(255))}
		if err != nil {
			panic(err)
		}
		p.Add(s)
	}
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "points.png"); err != nil {
		panic(err)
	}
}
