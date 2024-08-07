package accuracy

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type RegressionAccuracy struct {
	BaseAccuracy
	Precision float64 `json:"Precision"`
}

func (r *RegressionAccuracy) Initialization(target *mat.Dense) {
	precision := stat.StdDev(target.RawMatrix().Data, nil) / 250.0
	r.Precision = precision
}

func (r *RegressionAccuracy) Compare(predictions *mat.Dense, target *mat.Dense) [][]bool {
	rows, cols := predictions.Dims()

	result := make([][]bool, rows)

	for i := range result {
		result[i] = make([]bool, cols)
		for j := range result[i] {
			result[i][j] = math.Abs(predictions.At(i, j)-target.At(i, j)) < r.Precision
		}
	}

	return result
}
