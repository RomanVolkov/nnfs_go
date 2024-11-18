package marshaling

import (
	"encoding/json"

	"gonum.org/v1/gonum/mat"
	"main/layer"
	"reflect"
)

type ConvolutionWrapper struct {
	layer.ConvolutionLayer
}

type convolutionWrapperData struct {
	Depths      int               `json:"depths"`
	InputShape  layer.InputShape  `json:"input_shape"`
	OutputShape layer.InputShape  `json:"output_shape"`
	KernelShape layer.KernelShape `json:"kernel_shape"`
	KernelSize  int               `json:"kenel_size"`
	Kernels     [][]DenseWrapper  `json:"kernels"`
	Biases      []DenseWrapper    `json:"biases"`

	Type string `json:"type"`
}

func (value ConvolutionWrapper) MarshalJSON() ([]byte, error) {
	kernels := make([][]DenseWrapper, len(value.Kernels))
	for i := 0; i < len(value.Kernels); i++ {
		kernels[i] = make([]DenseWrapper, len(value.Kernels[i]))
		for j := 0; j < len(kernels[i]); j++ {
			kernels[i][j] = DenseWrapper{Dense: value.Kernels[i][j]}
		}
	}
	biases := make([]DenseWrapper, len(value.Biases))
	for i := 0; i < len(biases); i++ {
		biases[i] = DenseWrapper{Dense: value.Biases[i]}
	}

	layerData := convolutionWrapperData{
		Depths:      value.Depths,
		InputShape:  value.InputShape,
		OutputShape: value.OutputShape,
		KernelSize:  value.KernelSize,
		KernelShape: value.KernelShape,
		Kernels:     kernels,
		Biases:      biases,
	}

	typeStr := reflect.TypeOf(value.ConvolutionLayer).String()
	data := struct {
		Type string                 `json:"type"`
		Data convolutionWrapperData `json:"data"`
	}{
		Type: typeStr,
		Data: layerData,
	}

	return json.Marshal(data)
}

func (value *ConvolutionWrapper) UnmarshalJSON(data []byte) error {
	wrap := struct {
		Type string                 `json:"type"`
		Data convolutionWrapperData `json:"data"`
	}{}

	err := json.Unmarshal(data, &wrap)
	if err != nil {
		return err
	}
	l := layer.ConvolutionLayer{}
	l.Depths = wrap.Data.Depths
	l.InputShape = wrap.Data.InputShape
	l.OutputShape = wrap.Data.OutputShape
	l.KernelSize = wrap.Data.KernelSize
	l.KernelShape = wrap.Data.KernelShape

	l.Biases = make([]mat.Dense, len(wrap.Data.Biases))
	for i := 0; i < len(l.Biases); i++ {
		l.Biases[i] = wrap.Data.Biases[i].Dense
	}

	l.Kernels = make([][]mat.Dense, len(wrap.Data.Kernels))
	for i := 0; i < len(l.Kernels); i++ {
		l.Kernels[i] = make([]mat.Dense, len(wrap.Data.Kernels[i]))
		for j := 0; j < len(l.Kernels[i]); j++ {
			l.Kernels[i][j] = wrap.Data.Kernels[i][j].Dense
		}
	}

	value.ConvolutionLayer = l
	return nil
}
