package model

import (
	"fmt"
	"main/layer"
	"main/loss"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	inputLayer   layer.InputLayer
	layers       []layer.LayerInterface
	lossFunction loss.LossInterface
	optimizer    OptimizerInterface
}

func (m *Model) Add(layer layer.LayerInterface) {
	m.layers = append(m.layers, layer)
}

func (m *Model) Set(loss loss.LossInterface, optimizer OptimizerInterface) {
	m.lossFunction = loss
	m.optimizer = optimizer
}

func (m *Model) Train(x, y mat.Dense, epochs int, printEvery int) {
	for epoch := 0; epoch < epochs+1; epoch++ {
		output := m.Forward(x)
		fmt.Println(mat.Formatted(output))
	}
}

func (m *Model) Description() {
	for _, l := range m.layers {
		fmt.Print(l.Name(), ", ")
	}
	fmt.Println()
	fmt.Println(m.lossFunction.Name())
	fmt.Println(m.optimizer.Name())
}

func (m *Model) Finalize() {
	m.inputLayer = layer.InputLayer{}

	trainableLayers := make([]*layer.Layer, 0)
	for _, item := range m.layers {
		layer, ok := item.(*layer.Layer)
		if ok {
			trainableLayers = append(trainableLayers, layer)
		}
	}
	m.lossFunction.SetLayers(trainableLayers)
}

func (m *Model) Forward(input mat.Dense) *mat.Dense {
	m.inputLayer.Forward(&input)

	for i, layer := range m.layers {
		if i == 0 {
			layer.Forward(m.inputLayer.GetOutput())
		} else {
			layer.Forward(m.layers[i-1].GetOutput())
		}
	}

	return m.layers[len(m.layers)-1].GetOutput()
}
