package deep

import "math"

// Mode denotes inference mode
type Mode int

const (
	// ModeDefault is unspecified mode
	ModeDefault Mode = 0
	// ModeMultiClass is for one-hot encoded classification, applies softmax output layer
	ModeMultiClass Mode = 1
	// ModeRegression is regression, applies linear output layer
	ModeRegression Mode = 2
	// ModeBinary is binary classification, applies sigmoid output layer
	ModeBinary Mode = 3
	// ModeMultiLabel is for multilabel classification, applies sigmoid output layer
	ModeMultiLabel Mode = 4
)

// OutputActivation returns activation corresponding to prediction mode
func OutputActivation(c Mode) ActivationType {
	switch c {
	case ModeMultiClass:
		return ActivationSoftmax
	case ModeRegression:
		return ActivationLinear
	case ModeBinary, ModeMultiLabel:
		return ActivationSigmoid
	}
	return ActivationNone
}

// GetActivation returns the concrete activation given an ActivationType
func GetActivation(act ActivationType) Differentiable {
	switch act {
	case ActivationSigmoid:
		return Sigmoid{}
	case ActivationTanh:
		return Tanh{}
	case ActivationReLU:
		return ReLU{}
	case ActivationLinear:
		return Linear{}
	case ActivationSoftmax:
		return Softmax{}
	case ActivationLeakyReLU:
		return LeakyReLU{}
	case ActivationELU:
		return ELU{}
	case ActivationSwish:
		return Swish{}
	}
	return Linear{}
}

// ActivationType represents a neuron activation function
type ActivationType int

const (
	// ActivationNone is no activation
	ActivationNone ActivationType = 0
	// ActivationSigmoid is a sigmoid activation
	ActivationSigmoid ActivationType = 1
	// ActivationTanh is hyperbolic activation
	ActivationTanh ActivationType = 2
	// ActivationReLU is rectified linear unit activation
	ActivationReLU ActivationType = 3
	// ActivationLinear is linear activation
	ActivationLinear ActivationType = 4
	// ActivationSoftmax is a softmax activation (per layer)
	ActivationSoftmax ActivationType = 5
	// ActivationLeakyReLU is a leaky rectified linear unit activation
	ActivationLeakyReLU ActivationType = 6
	// ActivationELU is an exponential linear unit activation
	ActivationELU ActivationType = 7
	// ActivationSwish is a swish activation
	ActivationSwish ActivationType = 8
)

// Differentiable is an activation function and its first-order derivative,
// where the latter is expressed as a function of the former for efficiency
type Differentiable interface {
	F(float64) float64
	Df(float64) float64
}

// Sigmoid is a logistic activator in the special case of a = 1
type Sigmoid struct{}

// F is Sigmoid(x)
func (a Sigmoid) F(x float64) float64 { return Logistic(x, 1) }

// Df is Sigmoid'(y), where y = Sigmoid(x)
func (a Sigmoid) Df(y float64) float64 { return y * (1 - y) }

// Logistic is the logistic function
func Logistic(x, a float64) float64 {
	return 1 / (1 + math.Exp(-a*x))
}

// Tanh is a hyperbolic activator
type Tanh struct{}

// F is Tanh(x)
func (a Tanh) F(x float64) float64 { return (1 - math.Exp(-2*x)) / (1 + math.Exp(-2*x)) }

// Df is Tanh'(y), where y = Tanh(x)
func (a Tanh) Df(y float64) float64 { return 1 - math.Pow(y, 2) }

// ReLU is a rectified linear unit activator
type ReLU struct{}

// F is ReLU(x)
func (a ReLU) F(x float64) float64 { return math.Max(x, 0) }

// Df is ReLU'(y), where y = ReLU(x)
func (a ReLU) Df(y float64) float64 {
	if y > 0 {
		return 1
	}
	return 0
}

// Linear is a linear activator
type Linear struct{}

// F is the identity function
func (a Linear) F(x float64) float64 { return x }

// Df is constant
func (a Linear) Df(x float64) float64 { return 1 }

// Softmax is a softmax activator
type Softmax struct{}

// F is Softmax(x), currently implemented as identity for simplicity
func (a Softmax) F(x float64) float64 { return x }

// Df is Softmax'(y), where y = Softmax(x), currently implemented as constant 1 for simplicity
func (a Softmax) Df(y float64) float64 { return 1 }

// LeakyReLU is a leaky rectified linear unit activator
type LeakyReLU struct{}

// F is LeakyReLU(x)
func (a LeakyReLU) F(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}

// Df is LeakyReLU'(y), where y = LeakyReLU(x)
func (a LeakyReLU) Df(y float64) float64 {
	if y > 0 {
		return 1
	}
	return 0.01
}

// ELU is an exponential linear unit activator
type ELU struct{}

// F is ELU(x)
func (a ELU) F(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * (math.Exp(x) - 1)
}

// Df is ELU'(y), where y = ELU(x)
func (a ELU) Df(y float64) float64 {
	if y > 0 {
		return 1
	}
	return y + 0.01
}

// Swish is a swish activator
type Swish struct{}

// F is Swish(x)
func (a Swish) F(x float64) float64 { return x * Sigmoid{}.F(x) }

// Df is Swish'(y), where y = Swish(x)
func (a Swish) Df(y float64) float64 {
	sig := Sigmoid{}.F(y)
	return sig + y*(1-sig)
}
