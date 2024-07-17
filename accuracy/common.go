package accuracy

import "gonum.org/v1/gonum/mat"

type AccuracyInterface interface {
	Initialization(target *mat.Dense)
	Compare(predictions *mat.Dense, target *mat.Dense) [][]bool
	AddAccumulated(accuracySum float64, count int64)
	CalculateAccumulatedAccuracy() float64
	ResetAccumulated()
}

type BaseAccuracy struct {
	accumulatedAccuracySum float64
	accumulatedCount       int64
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

func (accuracy *BaseAccuracy) AddAccumulated(accuracySum float64, count int64) {
	accuracy.accumulatedAccuracySum += accuracySum
	accuracy.accumulatedCount += count
}

func (accuracy *BaseAccuracy) CalculateAccumulatedAccuracy() float64 {
	return accuracy.accumulatedAccuracySum / float64(accuracy.accumulatedCount)
}

func (accuracy *BaseAccuracy) ResetAccumulated() {
	accuracy.accumulatedAccuracySum = 0.0
	accuracy.accumulatedCount = 0
}
