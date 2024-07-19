package model

import (
	"fmt"
	"main/accuracy"
	"main/activation"
	"main/layer"
	"main/loss"
	"main/optimizer"
	"main/utils"

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
	m.lossFunction.Backward(&output, &target)

	for k := range m.layers {
		i := len(m.layers) - 1 - k
		if i == len(m.layers)-1 {
			m.layers[i].Backward(m.lossFunction.GetDInputs())
		} else {
			m.layers[i].Backward(m.layers[i+1].GetDInputs())
		}
	}
}

func (m *Model) Train(trainingData ModelData, epochs int, batchSize *int, printEvery int, validationData *ModelData) {
	x, y := trainingData.X, trainingData.Y
	m.accuracy.Initialization(&y)

	// default value if batch size is nil
	trainSteps := 1
	validationSteps := 1

	// calculate steps based on data lenght and batch size
	if batchSize != nil {
		count, _ := trainingData.X.Dims()
		trainSteps = count / *batchSize
		if trainSteps**batchSize < count {
			trainSteps += 1
		}

		if validationData != nil {
			count, _ = validationData.X.Dims()
			validationSteps = count / *batchSize
			if validationSteps**batchSize < count {
				validationSteps += 1
			}
		}
	}

	for epoch := 0; epoch < epochs+1; epoch++ {
		fmt.Println("Epoch", epoch)

		m.lossFunction.ResetAccumulated()
		m.accuracy.ResetAccumulated()

		for _, step := range utils.MakeRange(trainSteps) {
			var batchX mat.Dense
			var batchY mat.Dense
			if batchSize == nil {
				batchX = x
				batchY = y
			} else {
				_, cols := x.Dims()
				batchX = *mat.DenseCopyOf(x.Slice(step**batchSize, (step+1)**batchSize, 0, cols-1))
				_, cols = y.Dims()
				batchY = *mat.DenseCopyOf(y.Slice(step**batchSize, (step+1)**batchSize, 0, cols-1))
			}

			output := m.Forward(batchX, true)

			dataLoss := loss.CalculateLoss(m.lossFunction, output, &batchY)
			regularizationLoss := m.lossFunction.RegularizationLoss()
			lossValue := dataLoss + regularizationLoss

			predictions := m.outputLayerActivation.Predictions(output)
			accuracy := accuracy.CalculateAccuracy(m.accuracy, &predictions, &batchY)

			// TODO: do I need this still?
			// or loss funciton holds refereces to actual data?
			m.passTrainableLayer()

			m.Backward(*output, batchY)

			m.optimizer.PreUpdate()
			for _, item := range m.layers {
				layer, ok := item.(*layer.Layer)
				if ok {
					m.optimizer.UpdateParams(layer)
				}
			}

			m.optimizer.PostUpdate()

			if step%printEvery == 0 || step == trainSteps-1 {
				fmt.Println("step:", step, "\n",
					"loss:", lossValue,
					"(data loss:", dataLoss, "reg loss:", regularizationLoss, ") ",
					"acc:", accuracy,
					"lr", m.optimizer.GetCurrentLearningRate())
			}
		}

		epochDataLoss := m.lossFunction.CalculateAccumulatedLoss()
		epochRegularisationLoss := m.lossFunction.RegularizationLoss()
		epochLoss := epochDataLoss + epochRegularisationLoss
		epochAccuracy := m.accuracy.CalculateAccumulatedAccuracy()
		fmt.Println("training, ",
			"loss:", epochLoss,
			"(data loss:", epochDataLoss, "reg loss:", epochRegularisationLoss, ") ",
			"acc:", epochAccuracy,
			"lr", m.optimizer.GetCurrentLearningRate())
	}

	if validationData != nil {
		X_val, Y_val := validationData.X, validationData.Y
		m.lossFunction.ResetAccumulated()
		m.accuracy.ResetAccumulated()

		for _, step := range utils.MakeRange(validationSteps) {

			var batchX mat.Dense
			var batchY mat.Dense
			if batchSize == nil {
				batchX = X_val
				batchY = Y_val
			} else {
				_, cols := x.Dims()
				batchX = *mat.DenseCopyOf(X_val.Slice(step**batchSize, (step+1)**batchSize, 0, cols-1))
				_, cols = y.Dims()
				batchY = *mat.DenseCopyOf(Y_val.Slice(step**batchSize, (step+1)**batchSize, 0, cols-1))
			}
			validationOutput := m.Forward(batchX, false)
			loss.CalculateLoss(m.lossFunction, validationOutput, &batchY)
			validationPredictions := m.outputLayerActivation.Predictions(validationOutput)
			accuracy.CalculateAccuracy(m.accuracy, &validationPredictions, &batchY)
		}
		valLoss := m.lossFunction.CalculateAccumulatedLoss()
		valAccuracy := m.accuracy.CalculateAccumulatedAccuracy()
		fmt.Println("validation: ", "loss:", valLoss, "accuracy:", valAccuracy)
	}
}
