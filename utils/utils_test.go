package utils_test

import (
	"main/utils"
	"testing"
)

func TestMaxValue(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	maxValue := utils.MaxValue(values, 1, 2, 2, 2, 3)

	if maxValue != 9 {
		t.Fatalf("Incorrect max value: %v", maxValue)
	}

	maxValue = utils.MaxValue(values, 0, 1, 0, 1, 3)

	if maxValue != 5 {
		t.Fatalf("Incorrect max value: %v", maxValue)
	}
}
