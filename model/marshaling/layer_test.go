package marshaling

import (
	"errors"
	"main/layer"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLayerMarshaling(t *testing.T) {
	weights := []float64{1.0, 2.0, 3.0, 0.4, 1e-5, 60.0, 7.0, 0.08, 0.009, 10}
	biases := []float64{0.0, 0.1, 0.3}
	l := layer.Layer{}
	l.LoadFromParams(mat.NewDense(2, 5, weights), mat.NewDense(1, 3, biases), layer.Regularizer{}, layer.Regularizer{Weight: 0.1, Bias: 2.0})

	wrap := LayerWrapper{Layer: l}
	d, err := wrap.MarshalJSON()
	if err != nil {
		t.Error(err)
	}

	t.Log(string(d))

	loadedWrap := LayerWrapper{}
	err = loadedWrap.UnmarshalJSON(d)
	if err != nil {
		t.Error(err)
	}

	if !IsEqual(weights, loadedWrap.Weights.RawMatrix().Data) {
		t.Error(errors.New("incorrect weights are loaded"))
	}
	if !IsEqual(biases, loadedWrap.Biases.RawMatrix().Data) {
		t.Error(errors.New("incorrect biases are loaded"))
	}

	if loadedWrap.Layer.L2.Bias != 2.0 {
		t.Error(errors.New("incorrect L2 is loaded"))
	}
}
