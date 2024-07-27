package model

import (
	"encoding/json"
	"fmt"
	"log"
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

	layers := make([]interface{}, len(model.Layers))
	for _, layer := range model.Layers {
		t := reflect.TypeOf(layer)

		layerData, err := json.Marshal(layer)
		if err != nil {
			log.Panic(err)
		}
		fmt.Println(layerData)

		l := struct {
			Type string      `json:"type"`
			Data interface{} `json:"data"`
		}{t.String(), layerData}
		d, err := json.Marshal(l)
		if err != nil {
			log.Panic(err)
		}
		layers = append(layers, d)
	}

	d, err := json.Marshal(layers)
	fmt.Println(string(d))

	// how to marshal custom sequence?
	// 	layers                []layer.LayerInterface
	// 	 loss
	// 	 optimizer
	// 	 accuracy

	return nil
}
func (provider *JSONModelDataProvider) Load(path string) (*Model, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// call Finalize after loading of the data
	return nil, nil
}
