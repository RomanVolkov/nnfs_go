package model

import (
	"fmt"
	"main/accuracy"
	"main/activation"
	"main/layer"
	"main/loss"
	"main/optimizer"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	inputLayer            layer.InputLayer
	layers                []layer.LayerInterface
	outputLayerActivation activation.ActivationInterface

	lossFunction loss.LossInterface
	optimizer    optimizer.OptimizerInterface
	accuracy     accuracy.AccuracyInterface
}

type ModelData struct {
	X, Y mat.Dense
}

func (m *Model) Add(layer layer.LayerInterface) {
	m.layers = append(m.layers, layer)
}

func (m *Model) Set(loss loss.LossInterface, optimizer optimizer.OptimizerInterface, accuracy accuracy.AccuracyInterface) {
	m.lossFunction = loss
	m.optimizer = optimizer
	m.accuracy = accuracy
}

func (m *Model) passTrainableLayer() {
	trainableLayers := make([]*layer.Layer, 0)
	for _, item := range m.layers {
		layer, ok := item.(*layer.Layer)
		if ok {
			trainableLayers = append(trainableLayers, layer)
		}
	}
	m.lossFunction.SetLayers(trainableLayers)
}

func (m *Model) Finalize() {
	m.inputLayer = layer.InputLayer{}

	m.passTrainableLayer()

	// TODO: check is it's referenced or copied?
	// if yes - is it an issue?
	lastLayer := m.layers[len(m.layers)-1]
	activation, ok := lastLayer.(activation.ActivationInterface)
	if ok {
		m.outputLayerActivation = activation
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

func (m *Model) Forward(input mat.Dense, isTraining bool) *mat.Dense {
	m.inputLayer.Forward(&input, isTraining)

	for i, layer := range m.layers {
		if i == 0 {
			layer.Forward(m.inputLayer.GetOutput(), isTraining)
		} else {
			layer.Forward(m.layers[i-1].GetOutput(), isTraining)
		}
	}

	return m.layers[len(m.layers)-1].GetOutput()
}

func (m *Model) Backward(output mat.Dense, target mat.Dense) {
	// TODO: fix target pass
	m.lossFunction.Backward(&output, target.RawMatrix().Data)

	for k := range m.layers {
		i := len(m.layers) - 1 - k
		if i == len(m.layers)-1 {
			m.layers[i].Backward(m.lossFunction.GetDInputs())
		} else {
			m.layers[i].Backward(m.layers[i+1].GetDInputs())
		}
	}
}

func (m *Model) Train(trainingData ModelData, epochs int, printEvery int, validationData *ModelData) {
	x, y := trainingData.X, trainingData.Y
	m.accuracy.Initialization(&y)

	for epoch := 0; epoch < epochs+1; epoch++ {
		output := m.Forward(x, true)

		// TODO: fix target pass
		dataLoss := loss.CalculateLoss(m.lossFunction, output, y.RawMatrix().Data)
		regularizationLoss := m.lossFunction.RegularizationLoss()
		lossValue := dataLoss + regularizationLoss

		predictions := m.outputLayerActivation.Predictions(output)
		accuracy := accuracy.CalculateAccuracy(m.accuracy, &predictions, &y)

		// TODO: do I need this still?
		// or loss funciton holds refereces to actual data?
		m.passTrainableLayer()

		m.Backward(*output, y)

		m.optimizer.PreUpdate()
		for _, item := range m.layers {
			layer, ok := item.(*layer.Layer)
			if ok {
				m.optimizer.UpdateParams(layer)
			}
		}

		m.optimizer.PostUpdate()

		if epoch%printEvery == 0 {
			fmt.Println("epoch:", epoch, "\n",
				"loss:", lossValue,
				"(data loss:", dataLoss, "regularization loss:", regularizationLoss, ") ",
				"accuracy:", accuracy,
				"learning rate:", m.optimizer.GetCurrentLearningRate())
		}
	}

	if validationData != nil {
		X_val, Y_val := validationData.X, validationData.Y
		validationOutput := m.Forward(X_val, false)
		// TODO: fix target pass
		validationLoss := loss.CalculateLoss(m.lossFunction, validationOutput, Y_val.RawMatrix().Data)
		validationPredictions := m.outputLayerActivation.Predictions(validationOutput)
		validationAccuracy := accuracy.CalculateAccuracy(m.accuracy, &validationPredictions, &y)

		fmt.Println("validation: ", "loss:", validationLoss, "accuracy:", validationAccuracy)
	}
}
