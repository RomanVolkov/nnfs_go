package accuracy

import "gonum.org/v1/gonum/mat"

type AccuracyInterface interface {
	Initialization(target *mat.Dense)
	Compare(predictions *mat.Dense, target *mat.Dense) [][]bool
}

func CalculateAccuracy(accuracy AccuracyInterface, predictions *mat.Dense, target *mat.Dense) float64 {
	comparisons := accuracy.Compare(predictions, target)

	count := 0
	countTrue := 0
	for _, row := range comparisons {
		for _, value := range row {

			if value {
				countTrue += 1
			}
			count += 1
		}
	}

	return float64(countTrue) / float64(count)
}
