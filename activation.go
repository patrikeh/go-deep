package deep

import "math"

type ActivationFunction func(float64) float64

// Assumes input to df is f(x)
type Activation struct {
	f, df ActivationFunction
}

var Tanh = Activation{
	f:  func(x float64) float64 { return (1 - math.Exp(-2*x)) / (1 + math.Exp(-2*x)) },
	df: func(y float64) float64 { return 1 - math.Pow(y, 2) },
}

var ReLU = Activation{ // Leaky ReLU
	f: func(x float64) float64 { return math.Max(x, 0) },
	df: func(y float64) float64 {
		if y > 0 {
			return 1
		}
		return 0.01
	},
}

var Sigmoid = Activation{
	f:  NewLogisticFunc(1),
	df: func(y float64) float64 { return y * (1 - y) },
}

func NewLogisticFunc(a float64) ActivationFunction {
	return func(x float64) float64 {
		return LogisticFunc(x, a)
	}
}
func LogisticFunc(x, a float64) float64 {
	return 1 / (1 + math.Exp(-a*x))
}

var Linear = Activation{
	f:  func(x float64) float64 { return x },
	df: func(x float64) float64 { return 1 },
}
