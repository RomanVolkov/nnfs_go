package model

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	layers       []LayerInterface
	lossFunction LossInterface
	optimizer    OptimizerInterface
}

func (m *Model) Add(layer LayerInterface) {
	m.layers = append(m.layers, layer)
}

func (m *Model) Set(loss LossInterface, optimizer OptimizerInterface) {
	m.lossFunction = loss
	m.optimizer = optimizer
}

func (m *Model) Train(x, y mat.Dense, epochs int, printEvery int) {
	for epoch := 0; epoch < epochs+1; epoch++ {
		// TODO: train
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
