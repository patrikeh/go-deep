package deep

import "fmt"

// Config defines the network topology, activations, losses etc
type Config struct {
	// Number of inputs
	Inputs int
	// Defines topology:
	// For instance, [5 3 3] signifies a network with two hidden layers
	// containing 5 and 3 nodes respectively, followed an output layer
	// containing 3 nodes.
	Layout []int
	// Activation functions: {ActivationTanh, ActivationReLU, ActivationSigmoid, ActivationLinear}
	Activation ActivationType
	// Solver modes: {ModeRegression, ModeBinary, ModeMultiClass, ModeMultiLabel}
	Mode Mode
	// Initializer for weights, required: {NewNormal(random, σ, μ), NewUniform(random, width, μ)}
	Weight WeightInitializer `json:"-"`
	// Loss functions: {LossCrossEntropy, LossBinaryCrossEntropy, LossMeanSquared}
	Loss LossType
	// Apply bias nodes
	Bias bool
}

func (c *Config) applyDefaults() {
	if c.Activation == ActivationNone {
		c.Activation = ActivationSigmoid
	}
	if c.Loss == LossNone {
		switch c.Mode {
		case ModeMultiClass:
			c.Loss = LossCrossEntropy
		case ModeBinary, ModeMultiLabel:
			c.Loss = LossBinaryCrossEntropy
		default:
			c.Loss = LossMeanSquared
		}
	}
}

func (c *Config) validate() error {
	invalid := func(format string, args ...interface{}) error {
		return fmt.Errorf("%s: %w", fmt.Sprintf(format, args...), ErrInvalidConfig)
	}

	if c.Inputs < 1 {
		return invalid("%d inputs, need at least 1", c.Inputs)
	}
	if len(c.Layout) == 0 {
		return invalid("layout is empty")
	}
	for i, size := range c.Layout {
		if size < 1 {
			return invalid("layer %d has %d neurons, need at least 1", i, size)
		}
	}
	if c.Weight == nil {
		return invalid("weight initializer is required, e.g. NewNormal(rand.New(rand.NewSource(0)), 1, 0)")
	}
	if c.Activation < ActivationSigmoid || c.Activation > ActivationSoftmax {
		return invalid("unknown activation %d", c.Activation)
	}
	if c.Mode < ModeDefault || c.Mode > ModeMultiLabel {
		return invalid("unknown mode %d", c.Mode)
	}
	if c.Activation == ActivationSoftmax && (c.Mode != ModeDefault || len(c.Layout) > 1) {
		return invalid("softmax is only supported in the output layer, which ModeMultiClass sets")
	}

	output := c.outputActivation()
	if output == ActivationSoftmax && c.Layout[len(c.Layout)-1] < 2 {
		return invalid("softmax needs at least 2 outputs")
	}
	if !lossSupports(c.Loss, output) {
		return invalid("loss %s can't be used with output activation %d", c.Loss, output)
	}
	return nil
}

func (c *Config) outputActivation() ActivationType {
	if c.Mode == ModeDefault {
		return c.Activation
	}
	return OutputActivation(c.Mode)
}

// lossSupports reports whether Loss.Df is the gradient of the loss for these outputs
func lossSupports(loss LossType, output ActivationType) bool {
	switch loss {
	case LossMeanSquared:
		return true
	case LossCrossEntropy:
		return output == ActivationSoftmax
	case LossBinaryCrossEntropy:
		return output == ActivationSigmoid
	}
	return false
}
