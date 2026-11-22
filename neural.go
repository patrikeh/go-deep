package deep

import (
	"errors"
	"fmt"
	"slices"
	"sync"
)

var (
	// ErrInvalidConfig means a network can't be built from a Config
	ErrInvalidConfig = errors.New("invalid config")
	// ErrDimension means a slice doesn't match the layer it is for
	ErrDimension = errors.New("dimension mismatch")
)

// Neural is a neural network. Predict is safe for concurrent use, training is not.
type Neural struct {
	Layers []*Layer
	Biases [][]*Synapse
	Config *Config

	mu sync.Mutex // guards neuron and synapse values during a forward pass
}

// NewNeural returns a new neural network. It fills in defaults on a copy of c, so c can be reused.
func NewNeural(c *Config) (*Neural, error) {
	if c == nil {
		return nil, fmt.Errorf("config is nil: %w", ErrInvalidConfig)
	}
	config := *c
	config.Layout = slices.Clone(c.Layout)
	config.applyDefaults()
	if err := config.validate(); err != nil {
		return nil, err
	}

	layers := initializeLayers(&config)

	var biases [][]*Synapse
	if config.Bias {
		biases = make([][]*Synapse, len(layers))
		for i, l := range layers {
			biases[i] = l.ApplyBias(config.Weight)
		}
	}

	return &Neural{
		Layers: layers,
		Biases: biases,
		Config: &config,
	}, nil
}

func initializeLayers(c *Config) []*Layer {
	layers := make([]*Layer, len(c.Layout))
	for i := range layers {
		act := c.Activation
		if i == (len(layers)-1) && c.Mode != ModeDefault {
			act = OutputActivation(c.Mode)
		}
		layers[i] = NewLayer(c.Layout[i], act)
	}

	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], c.Weight)
	}

	for _, neuron := range layers[0].Neurons {
		neuron.In = make([]*Synapse, c.Inputs)
		for i := range neuron.In {
			neuron.In[i] = NewSynapse(c.Weight())
		}
	}

	return layers
}

// Clone returns an independent network with the same topology, activations and weights
func (n *Neural) Clone() (*Neural, error) {
	config := *n.Config
	config.Weight = zeroWeight
	clone, err := NewNeural(&config)
	if err != nil {
		return nil, fmt.Errorf("cloning: %w", err)
	}
	// activations can be changed per neuron after construction
	for i, l := range n.Layers {
		for j, neuron := range l.Neurons {
			clone.Layers[i].Neurons[j].A = neuron.A
		}
	}
	if err := clone.ApplyWeights(n.Weights()); err != nil {
		return nil, fmt.Errorf("cloning: %w", err)
	}
	return clone, nil
}

func (n *Neural) fire() {
	for _, b := range n.Biases {
		for _, s := range b {
			s.fire(1)
		}
	}
	for _, l := range n.Layers {
		l.fire()
	}
}

// Forward computes a forward pass, leaving the result in each neuron's Value
func (n *Neural) Forward(input []float64) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.forward(input)
}

func (n *Neural) forward(input []float64) error {
	if len(input) != n.Config.Inputs {
		return fmt.Errorf("got %d inputs, want %d: %w", len(input), n.Config.Inputs, ErrDimension)
	}
	for _, n := range n.Layers[0].Neurons {
		for i := range input {
			n.In[i].fire(input[i])
		}
	}
	n.fire()
	return nil
}

// Predict computes a forward pass and returns a prediction
func (n *Neural) Predict(input []float64) ([]float64, error) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if err := n.forward(input); err != nil {
		return nil, fmt.Errorf("predicting: %w", err)
	}
	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]float64, len(outLayer.Neurons))
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Value
	}
	return out, nil
}

// NumWeights returns the number of weights in the network
func (n *Neural) NumWeights() (num int) {
	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			num += len(n.In)
		}
	}
	return
}

func (n *Neural) String() string {
	var s string
	for _, l := range n.Layers {
		s = fmt.Sprintf("%s\n%s", s, l)
	}
	return s
}
