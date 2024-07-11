package accuracy

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type CategorialAccuracy struct{}

func (r *CategorialAccuracy) Initialization(target *mat.Dense) {}

func (r *CategorialAccuracy) Compare(predictions *mat.Dense, target *mat.Dense) [][]bool {
	rows, _ := predictions.Dims()
	result := make([][]bool, rows)
	for i := range result {
		result[i] = make([]bool, 1)

		rowValues := predictions.RawRowView(i)
		predictedValue := floats.MaxIdx(rowValues)

		result[i][0] = predictedValue == int(target.At(i, 0))
	}

	return result
}
