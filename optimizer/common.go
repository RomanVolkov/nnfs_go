package optimizer

import "main/layer"

type OptimizerInterface interface {
	Name() string
	PreUpdate()
	UpdateParams(layer *layer.Layer)
	PostUpdate()
}
