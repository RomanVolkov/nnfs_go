package accuracy

import (
	"main/utils"

	"gonum.org/v1/gonum/mat"
)

type BinaryCategorialAccuracy struct {
	BaseAccuracy
}

func (r *BinaryCategorialAccuracy) Initialization(target *mat.Dense) {}

func (r *BinaryCategorialAccuracy) Compare(predictions *mat.Dense, target *mat.Dense) [][]bool {
	var p, t mat.Matrix = predictions, target
	if !utils.CompareDims(&p, &t) {
		panic("incorrect dimentions")
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
