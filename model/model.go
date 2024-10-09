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
	Layers    []layer.LayerInterface
	Loss      loss.LossInterface
	Optimizer optimizer.OptimizerInterface
	Accuracy  accuracy.AccuracyInterface

	inputLayer            layer.InputLayer
	outputLayerActivation activation.ActivationInterface
}

type ModelData struct {
	X, Y mat.Dense
}

func (m *Model) Add(layer layer.LayerInterface) {
	m.Layers = append(m.Layers, layer)
}

func (m *Model) Set(loss loss.LossInterface, optimizer optimizer.OptimizerInterface, accuracy accuracy.AccuracyInterface) {
	m.Loss = loss
	m.Optimizer = optimizer
	m.Accuracy = accuracy
}

func (m *Model) passTrainableLayer() {
	trainableLayers := make([]*layer.DenseLayer, 0)
	for _, item := range m.Layers {
		layer, ok := item.(*layer.DenseLayer)
		if ok {
			trainableLayers = append(trainableLayers, layer)
		}
	}
	m.Loss.SetLayers(trainableLayers)
}

func (m *Model) Finalize() {
	m.inputLayer = layer.InputLayer{}

	m.passTrainableLayer()

	// TODO: check is it's referenced or copied?
	// if yes - is it an issue?
	lastLayer := m.Layers[len(m.Layers)-1]
	activation, ok := lastLayer.(activation.ActivationInterface)
	if ok {
		m.outputLayerActivation = activation
	}
}

func (m *Model) Description() {
	for _, l := range m.Layers {
		fmt.Print(l.Name(), ", ")
	}
	fmt.Println()
	fmt.Println(m.Loss.Name())
	fmt.Println(m.Optimizer.Name())
}

func (m *Model) Forward(input mat.Dense, isTraining bool) *mat.Dense {
	m.inputLayer.Forward(&input, isTraining)

	for i, layer := range m.Layers {
		if i == 0 {
			layer.Forward(m.inputLayer.GetOutput(), isTraining)
		} else {
			layer.Forward(m.Layers[i-1].GetOutput(), isTraining)
		}
	}

	return m.Layers[len(m.Layers)-1].GetOutput()
}

func (m *Model) Backward(output mat.Dense, target mat.Dense) {
	m.Loss.Backward(&output, &target)

	for k := range m.Layers {
		i := len(m.Layers) - 1 - k
		if i == len(m.Layers)-1 {
			m.Layers[i].Backward(m.Loss.GetDInputs())
		} else {
			m.Layers[i].Backward(m.Layers[i+1].GetDInputs())
		}
	}
}

func (m *Model) Train(trainingData ModelData, epochs int, batchSize *int, printEvery int, validationData *ModelData) {
	fmt.Println("================================")
	fmt.Println("Training")
	m.Accuracy.Initialization(&trainingData.Y)

	// default value if batch size is nil
	trainSteps := 1

	// calculate steps based on data lenght and batch size
	if batchSize != nil {
		trainSteps = calculateSteps(trainingData, *batchSize)
	}

	for epoch := 0; epoch < epochs+1; epoch++ {
		fmt.Println("Epoch", epoch)

		m.Loss.ResetAccumulated()
		m.Accuracy.ResetAccumulated()

		for _, step := range utils.MakeRange(trainSteps) {
			batchX, batchY := makeBatch(trainingData, step, batchSize)
			output := m.Forward(batchX, true)

			dataLoss := loss.CalculateLoss(m.Loss, output, &batchY)
			regularizationLoss := m.Loss.RegularizationLoss()
			lossValue := dataLoss + regularizationLoss

			predictions := m.outputLayerActivation.Predictions(output)
			accuracy := accuracy.CalculateAccuracy(m.Accuracy, &predictions, &batchY)

			// TODO: do I need this still?
			// or loss funciton holds refereces to actual data?
			m.passTrainableLayer()

			m.Backward(*output, batchY)

			m.Optimizer.PreUpdate()
			for _, item := range m.Layers {
				layer, ok := item.(*layer.DenseLayer)
				if ok {
					m.Optimizer.UpdateParams(layer)
				}
			}

			m.Optimizer.PostUpdate()

			if step%printEvery == 0 || step == trainSteps-1 {
				fmt.Println("step:", step, "\n",
					"loss:", lossValue,
					"(data loss:", dataLoss, "reg loss:", regularizationLoss, ") ",
					"acc:", accuracy,
					"lr", m.Optimizer.GetCurrentLearningRate())
			}
		}

		epochDataLoss := m.Loss.CalculateAccumulatedLoss()
		epochRegularisationLoss := m.Loss.RegularizationLoss()
		epochLoss := epochDataLoss + epochRegularisationLoss
		epochAccuracy := m.Accuracy.CalculateAccumulatedAccuracy()
		fmt.Println("training, ",
			"loss:", epochLoss,
			"(data_loss:", epochDataLoss, "reg_loss:", epochRegularisationLoss, ") ",
			"acc:", epochAccuracy,
			"lr", m.Optimizer.GetCurrentLearningRate())
	}

	if validationData != nil {
		m.Evaluate(*validationData, batchSize)
	}
}

func (m *Model) Evaluate(data ModelData, batchSize *int) {
	fmt.Println("================================")
	fmt.Println("Evaluation")
	validationSteps := 1
	if batchSize != nil {
		validationSteps = calculateSteps(data, *batchSize)
	}

	m.Loss.RegularizationLoss()
	m.Accuracy.ResetAccumulated()

	for _, step := range utils.MakeRange(validationSteps) {
		batchX, batchY := makeBatch(data, step, batchSize)
		validationOutput := m.Forward(batchX, false)

		loss.CalculateLoss(m.Loss, validationOutput, &batchY)

		validationPredictions := m.outputLayerActivation.Predictions(validationOutput)
		accuracy.CalculateAccuracy(m.Accuracy, &validationPredictions, &batchY)
	}
	valLoss := m.Loss.CalculateAccumulatedLoss()
	valAccuracy := m.Accuracy.CalculateAccumulatedAccuracy()
	fmt.Println("validation:", "loss:", valLoss, "accuracy:", valAccuracy)
}

func (m *Model) Predict(inputSamples *mat.Dense, batchSize *int) mat.Dense {
	predictionSteps := 1
	if batchSize != nil {
		predictionSteps = calculateSteps(ModelData{X: *inputSamples, Y: *inputSamples}, *batchSize)
	}

	var output *mat.Dense
	for _, step := range utils.MakeRange(predictionSteps) {
		batchX, _ := makeBatch(ModelData{X: *inputSamples, Y: *inputSamples}, step, batchSize)
		validationOutput := m.Forward(batchX, false)

		if output == nil {
			// init output from first response
			output = mat.DenseCopyOf(validationOutput)
		} else {
			// append more outputs
			oR, oC := output.Dims()
			nR, _ := validationOutput.Dims()
			tmp := mat.NewDense(oR+nR, oC, nil)
			tmp.Stack(output, validationOutput)
			output = tmp
		}

	}

	return *output
}

func calculateSteps(data ModelData, batchSize int) int {
	count, _ := data.X.Dims()
	steps := count / batchSize
	if steps*batchSize < count {
		steps += 1
	}
	return steps
}

func makeBatch(data ModelData, step int, batchSize *int) (mat.Dense, mat.Dense) {
	var batchX mat.Dense
	var batchY mat.Dense
	if batchSize == nil {
		batchX = data.X
		batchY = data.Y
	} else {
		rows, cols := data.X.Dims()
		targetRowIndex := (step + 1) * *batchSize
		if targetRowIndex > rows {
			targetRowIndex = rows - 1
		}
		batchX = *mat.DenseCopyOf(data.X.Slice(step**batchSize, targetRowIndex, 0, cols))

		rows, cols = data.Y.Dims()
		targetRowIndex = (step + 1) * *batchSize
		if targetRowIndex > rows {
			targetRowIndex = rows - 1
		}
		batchY = *mat.DenseCopyOf(data.Y.Slice(step**batchSize, targetRowIndex, 0, cols))
	}
	return batchX, batchY
}
