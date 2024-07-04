package optimizer

import "main/layer"

type OptimizerInterface interface {
	Name() string
	GetCurrentLearningRate() float64
	PreUpdate()
	UpdateParams(layer *layer.Layer)
	PostUpdate()
}
