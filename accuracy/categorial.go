package accuracy

import (
	"main/utils"

	"gonum.org/v1/gonum/mat"
)

type CategorialAccuracy struct{}

func (r *CategorialAccuracy) Initialization(target *mat.Dense) {}

func (r *CategorialAccuracy) Compare(predictions *mat.Dense, target *mat.Dense) [][]bool {
	if !utils.CompareDims(predictions, target) {
		panic("inccorect dimentions")
	}

	rows, cols := predictions.Dims()
	result := make([][]bool, rows)
	for i := range result {
		result[i] = make([]bool, cols)

		for j := range result[i] {
			// ==============================================================
			// assumption is: predictions at this moment has categories values
			// ==============================================================
			result[i][j] = int64(predictions.At(i, j)) == int64(target.At(i, j))
		}
	}

	return result
}
