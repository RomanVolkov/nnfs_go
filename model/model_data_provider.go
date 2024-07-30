package model

import (
	"encoding/json"
	"errors"
	"io"
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
		var l *layer.Layer = &layer.Layer{}
		var relu *activation.Activation_ReLU = &activation.Activation_ReLU{}
		var softmax *activation.SoftmaxActivation = &activation.SoftmaxActivation{}

		if reflect.TypeOf(item).String() == reflect.TypeOf(l).String() {
			l, _ := item.(*layer.Layer)
			layersWraps = append(layersWraps, marshaling.LayerWrapper{Layer: *l})
		}
		if reflect.TypeOf(item).String() == reflect.TypeOf(relu).String() {
			v := struct {
				Type string `json:"type"`
			}{
				Type: reflect.TypeOf(item).String(),
			}
			layersWraps = append(layersWraps, v)
		}
		if reflect.TypeOf(item).String() == reflect.TypeOf(softmax).String() {
			v := struct {
				Type string `json:"type"`
			}{
				Type: reflect.TypeOf(item).String(),
			}
			layersWraps = append(layersWraps, v)
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
		Layers    []interface{} `json:"layers"`
		Loss      string        `json:"loss"`
		Accuracy  string        `json:"accuracy"`
		Optimizer interface{}   `json:"optimizer"`
	}{
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

	layers, ok := dict["layers"].([]interface{})
	if ok {
		for _, l := range layers {
			layerData, ok := l.(map[string]interface{})
			if ok {
				// decode layer.Layer
				if layerData["type"] == reflect.TypeOf(layer.Layer{}).String() {
					layerWrap := marshaling.LayerWrapper{}
					bd, err := json.Marshal(l)
					if err != nil {
						return nil, err
					}
					err = json.Unmarshal(bd, &layerWrap)
					layer := layer.Layer{}
					layer.LoadFromParams(&layerWrap.Weights, &layerWrap.Biases, layerWrap.L1, layerWrap.L2)
					m.Add(&layer)
				}
				// decode activation.Activation_ReLU
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
	default:
		return nil, errors.New("unsupported optimizer")
	}

	m.Set(lossValue, optimizerValue, accuracyValue)
	m.Finalize()
	m.Description()

	return &m, nil
}
