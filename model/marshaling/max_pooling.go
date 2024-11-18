package marshaling

import "main/layer"
import "reflect"
import "encoding/json"

type MaxPoolingWrapper struct {
	layer.MaxPoolingLayer
}

type maxPoolingWrapperData struct {
	PoolSize    int              `json:"pool_size"`
	InputShape  layer.InputShape `json:"input_shape"`
	OutputShape layer.InputShape `json:"output_shape"`
}

func (value MaxPoolingWrapper) MarshalJSON() ([]byte, error) {
	layerData := maxPoolingWrapperData{
		PoolSize:    value.PoolSize,
		InputShape:  value.InputShape,
		OutputShape: value.OutputShape,
	}

	typeStr := reflect.TypeOf(value.MaxPoolingLayer).String()
	data := struct {
		Type string                `json:"type"`
		Data maxPoolingWrapperData `json:"data"`
	}{
		Type: typeStr,
		Data: layerData,
	}

	return json.Marshal(data)
}

func (value *MaxPoolingWrapper) UnmarshalJSON(data []byte) error {
	wrap := struct {
		Type string                `json:"type"`
		Data maxPoolingWrapperData `json:"data"`
	}{}

	err := json.Unmarshal(data, &wrap)
	if err != nil {
		return err
	}

	l := layer.MaxPoolingLayer{}
	l.PoolSize = wrap.Data.PoolSize
	l.InputShape = wrap.Data.InputShape
	l.OutputShape = wrap.Data.OutputShape

	value.MaxPoolingLayer = l
	return nil
}
