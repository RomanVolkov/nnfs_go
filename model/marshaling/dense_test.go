package marshaling

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDenseMarshaling(t *testing.T) {
	arr := []float64{10.0, 1e-5, 0.1}
	testValue := DenseWrapper{Dense: *mat.NewDense(1, 3, arr)}

	d, err := testValue.MarshalJSON()
	if err != nil {
		t.Error(err)
	}

	loaded := DenseWrapper{}
	err = loaded.UnmarshalJSON(d)
	if err != nil {
		t.Error(err)
	}
	if !IsEqual(arr, loaded.RawMatrix().Data) {
		t.Error("different values")
		fmt.Println(arr)
		fmt.Println(loaded.Dense.RawMatrix().Data)
	}
}

func IsEqual(s1, s2 []float64) bool {
	if len(s1) != len(s2) {
		return false
	}

	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}

	return true
}
