package model

import (
	"encoding/json"
	"errors"
	"io"
	"log"
	"main/accuracy"
	"main/activation"
	"main/layer"
	"main/loss"
	"main/model/marshaling"
	"main/optimizer"
	"os"
	"reflect"
)

type ModelDataProvider interface {
	Store(path string, model *Model) error
	Load(path string) (*Model, error)
}

type JSONModelDataProvider struct {
}

func (provider *JSONModelDataProvider) Store(path string, model *Model) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	layersWraps := make([]interface{}, 0)
	for _, item := range model.Layers {
		var l *layer.DenseLayer = &layer.DenseLayer{}
		var relu *activation.Activation_ReLU = &activation.Activation_ReLU{}
		var softmax *activation.SoftmaxActivation = &activation.SoftmaxActivation{}
		var sigmoid *activation.SigmoidActivation = &activation.SigmoidActivation{}
		var convolution *layer.ConvolutionLayer = &layer.ConvolutionLayer{}
		var maxPooling *layer.MaxPoolingLayer = &layer.MaxPoolingLayer{}

		if reflect.TypeOf(item).String() == reflect.TypeOf(l).String() {
			l, _ := item.(*layer.DenseLayer)
			layersWraps = append(layersWraps, marshaling.LayerWrapper{DenseLayer: *l})
		} else if reflect.TypeOf(item).String() == reflect.TypeOf(convolution).String() {
			convolutionLayer, _ := item.(*layer.ConvolutionLayer)
			layersWraps = append(layersWraps, marshaling.ConvolutionWrapper{ConvolutionLayer: *convolutionLayer})
		} else if reflect.TypeOf(item).String() == reflect.TypeOf(relu).String() {
			v := struct {
				Type string `json:"type"`
			}{
				Type: reflect.TypeOf(item).String(),
			}
			layersWraps = append(layersWraps, v)
		} else if reflect.TypeOf(item).String() == reflect.TypeOf(softmax).String() {
			v := struct {
				Type string `json:"type"`
			}{
				Type: reflect.TypeOf(item).String(),
			}
			layersWraps = append(layersWraps, v)
		} else if reflect.TypeOf(item).String() == reflect.TypeOf(sigmoid).String() {
			v := struct {
				Type string `json:"type"`
			}{
				Type: reflect.TypeOf(item).String(),
			}
			layersWraps = append(layersWraps, v)
		} else if reflect.TypeOf(item).String() == reflect.TypeOf(maxPooling).String() {
			maxPoolingLayer, _ := item.(*layer.MaxPoolingLayer)
			layersWraps = append(layersWraps, marshaling.MaxPoolingWrapper{MaxPoolingLayer: *maxPoolingLayer})
		} else {
			log.Fatalf("Unknown type: %v", reflect.TypeOf(item).String())
		}
	}

	o := struct {
		Type string      `json:"type"`
		Data interface{} `json:"data"`
	}{
		Type: reflect.TypeOf(model.Optimizer).String(),
		Data: model.Optimizer,
	}

	root := struct {
		Name      string        `json:"name"`
		Layers    []interface{} `json:"layers"`
		Loss      string        `json:"loss"`
		Accuracy  string        `json:"accuracy"`
		Optimizer interface{}   `json:"optimizer"`
	}{
		Name:      model.Name,
		Layers:    layersWraps,
		Loss:      reflect.TypeOf(model.Loss).String(),
		Accuracy:  reflect.TypeOf(model.Accuracy).String(),
		Optimizer: o,
	}

	d, err := json.Marshal(root)

	_, err = file.Write(d)
	if err != nil {
		return err
	}

	return nil
}
func (provider *JSONModelDataProvider) Load(path string) (*Model, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	dict := map[string]interface{}{}
	json.Unmarshal(data, &dict)

	m := Model{}
	name, _ := dict["name"].(string)
	m.Name = name

	layers, ok := dict["layers"].([]interface{})
	if ok {
		for _, l := range layers {
			layerData, ok := l.(map[string]interface{})
			if ok {
				// decode layer.Layer
				if layerData["type"] == reflect.TypeOf(layer.DenseLayer{}).String() {
					layerWrap := marshaling.LayerWrapper{}
					bd, err := json.Marshal(l)
					if err != nil {
						return nil, err
					}
					err = json.Unmarshal(bd, &layerWrap)
					layer := layer.DenseLayer{}
					layer.LoadFromParams(&layerWrap.Weights, &layerWrap.Biases, layerWrap.L1, layerWrap.L2)
					m.Add(&layer)
				}

				// decode layer.ConvolutionLayer
				if layerData["type"] == reflect.TypeOf(layer.ConvolutionLayer{}).String() {
					layerWrap := marshaling.ConvolutionWrapper{}
					bd, err := json.Marshal(l)
					if err != nil {
						return nil, err
					}
					err = json.Unmarshal(bd, &layerWrap)
					layer := layer.ConvolutionLayer{}
					layer.LoadFromParams(layerWrap.InputShape, layerWrap.Depths, layerWrap.KernelSize, layerWrap.OutputShape, layerWrap.KernelShape, layerWrap.Kernels, layerWrap.Biases)
					m.Add(&layer)
				}
				// decode activation.Activation_ReLU

				// decode layer.MaxPoolingLayer
				if layerData["type"] == reflect.TypeOf(layer.MaxPoolingLayer{}).String() {
					layerWrap := marshaling.MaxPoolingWrapper{}
					bd, err := json.Marshal(l)
					if err != nil {
						return nil, err
					}
					err = json.Unmarshal(bd, &layerWrap)
					layer := layer.MaxPoolingLayer{}
					layer.LoadFromParams(layerWrap.PoolSize, layerWrap.InputShape, layerWrap.OutputShape)
					m.Add(&layer)
				}

				var relu *activation.Activation_ReLU = &activation.Activation_ReLU{}
				if layerData["type"] == reflect.TypeOf(relu).String() {
					a := activation.Activation_ReLU{}
					m.Add(&a)
				}

				// decode SoftmaxActivation
				var softmax *activation.SoftmaxActivation = &activation.SoftmaxActivation{}
				if layerData["type"] == reflect.TypeOf(softmax).String() {
					a := activation.SoftmaxActivation{}
					m.Add(&a)
				}

				// decode SigmoidActivation
				var sigmoid *activation.SigmoidActivation = &activation.SigmoidActivation{}
				if layerData["type"] == reflect.TypeOf(sigmoid).String() {
					a := activation.SigmoidActivation{}
					m.Add(&a)
				}
			} else {
				return nil, errors.New("failed to typecase layerData")
			}
		}
	} else {
		return nil, errors.New("failed to get layers")
	}

	lossString, _ := dict["loss"].(string)
	var lossValue loss.LossInterface
	switch lossString {
	case reflect.TypeOf(&loss.CategoricalCrossentropyLoss{}).String():
		lossValue = &loss.CategoricalCrossentropyLoss{}
	default:
		return nil, errors.New("Unsupported loss")
	}

	accuracyString, _ := dict["accuracy"].(string)
	var accuracyValue accuracy.AccuracyInterface
	switch accuracyString {
	case reflect.TypeOf(&accuracy.CategorialAccuracy{}).String():
		accuracyValue = &accuracy.CategorialAccuracy{}
	default:
		return nil, errors.New("Unsupported loss")
	}

	var optimizerValue optimizer.OptimizerInterface
	optimizerDict, ok := dict["optimizer"].(map[string]interface{})
	if !ok {
		return nil, errors.New("cannot get optimizer data")
	}
	optimizerType := optimizerDict["type"].(string)
	switch optimizerType {
	case reflect.TypeOf(&optimizer.OptimizerAdam{}).String():
		d, err := json.Marshal(optimizerDict["data"])
		if err != nil {
			return nil, err
		}
		wrap := optimizer.OptimizerAdam{}
		err = json.Unmarshal(d, &wrap)
		if err != nil {
			return nil, err
		}
		optimizerValue = &wrap
	case reflect.TypeOf(&optimizer.OptimizerSGD{}).String():
		d, err := json.Marshal(optimizerDict["data"])
		if err != nil {
			return nil, err
		}
		wrap := optimizer.OptimizerSGD{}
		err = json.Unmarshal(d, &wrap)
		if err != nil {
			return nil, err
		}
		optimizerValue = &wrap
	default:
		return nil, errors.New("unsupported optimizer")
	}

	m.Set(lossValue, optimizerValue, accuracyValue)
	m.Finalize()
	m.Description()

	return &m, nil
}
