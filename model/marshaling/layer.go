package marshaling

import (
	"encoding/json"
	"main/layer"
	"reflect"
)

type LayerWrapper struct {
	layer.Layer
}

type regularizerWrapper struct {
	Weight float64 `json:"weight"`
	Bias   float64 `json:"bias"`
}
type layerWrapperData struct {
	Weights DenseWrapper       `json:"weights"`
	Biases  DenseWrapper       `json:"biases"`
	L1      regularizerWrapper `json:"l1"`
	L2      regularizerWrapper `json:"l2"`
	Type    string             `json:"type"`
}

func (value LayerWrapper) MarshalJSON() ([]byte, error) {
	layerData := layerWrapperData{
		Weights: DenseWrapper{Dense: value.Weights},
		Biases:  DenseWrapper{Dense: value.Biases},
		L1:      regularizerWrapper{Weight: value.L1.Weight, Bias: value.L1.Bias},
		L2:      regularizerWrapper{Weight: value.L2.Weight, Bias: value.L2.Bias},
		Type:    reflect.TypeOf(value.Layer).String(),
	}
	typeStr := reflect.TypeOf(value.Layer).String()
	data := struct {
		Type string           `json:"type"`
		Data layerWrapperData `json:"data"`
	}{
		Type: typeStr,
		Data: layerData,
	}

	return json.Marshal(data)
}

func (value *LayerWrapper) UnmarshalJSON(data []byte) error {
	wrap := struct {
		Type string           `json:"type"`
		Data layerWrapperData `json:"data"`
	}{}

	err := json.Unmarshal(data, &wrap)
	if err != nil {
		return err
	}

	l := layer.Layer{}
	l.Weights = wrap.Data.Weights.Dense
	l.Biases = wrap.Data.Biases.Dense
	l.L1 = layer.Regularizer{Weight: wrap.Data.L1.Weight, Bias: wrap.Data.L1.Bias}
	l.L2 = layer.Regularizer{Weight: wrap.Data.L2.Weight, Bias: wrap.Data.L2.Bias}
	value.Layer = l
	return nil
}
