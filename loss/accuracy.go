package loss

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func CalculateAccuracy(prediction *mat.Dense, targetClasses []uint8) float64 {
	predictionClasses := make([]uint8, len(targetClasses))
	r, _ := prediction.Dims()
	for i := 0; i < r; i++ {
		rowValues := prediction.RawRowView(i)
		maxIndex := floats.MaxIdx(rowValues)
		predictionClasses[i] = uint8(maxIndex)
	}

	sum := 0
	for i := range predictionClasses {
		if predictionClasses[i] == targetClasses[i] {
			sum += 1
		}
	}
	return float64(sum) / float64(len(targetClasses))
}
