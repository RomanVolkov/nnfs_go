package loss

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
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

func CalculateBinaryAccuracy(predicitons *mat.Dense, targetClasses []uint8) float64 {
	// assume predicitons len is equal to targetClasses
	pred := make([]uint8, len(predicitons.RawMatrix().Data))
	for i, v := range predicitons.RawMatrix().Data {
		if v > 0.5 {
			pred[i] = 1
		} else {
			pred[i] = 0
		}
	}

	count := 0
	for i := range pred {
		if pred[i] == targetClasses[i] {
			count += 1
		}
	}
	return float64(count) / float64(len(pred))
}

func CalculateRegressionAccuracy(predictions *mat.Dense, targetValues []float64) float64 {
	// assume predictions are 1D array
	precision := stat.StdDev(predictions.RawMatrix().Data, nil) / 250.0
	preds := predictions.RawMatrix().Data

	count := 0

	for i := range preds {
		if math.Abs(preds[i]-targetValues[i]) < precision {
			count += 1
		}
	}

	return float64(count) / float64(len(preds))
}
