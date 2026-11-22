package deep

import (
	"encoding/json"
	"fmt"
	"slices"
)

// Dump is a neural network dump
type Dump struct {
	Config  *Config
	Weights [][][]float64
}

// ApplyWeights sets the weights from a three-dimensional slice, shaped as Weights returns them
func (n *Neural) ApplyWeights(weights [][][]float64) error {
	if err := n.checkShape(weights); err != nil {
		return fmt.Errorf("applying weights: %w", err)
	}
	for i, l := range n.Layers {
		for j, neuron := range l.Neurons {
			for k, s := range neuron.In {
				s.Weight = weights[i][j][k]
			}
		}
	}
	return nil
}

func (n *Neural) checkShape(weights [][][]float64) error {
	if len(weights) != len(n.Layers) {
		return fmt.Errorf("got %d layers, want %d: %w", len(weights), len(n.Layers), ErrDimension)
	}
	for i, l := range n.Layers {
		if len(weights[i]) != len(l.Neurons) {
			return fmt.Errorf("layer %d: got %d neurons, want %d: %w", i, len(weights[i]), len(l.Neurons), ErrDimension)
		}
		for j, neuron := range l.Neurons {
			if len(weights[i][j]) != len(neuron.In) {
				return fmt.Errorf("layer %d, neuron %d: got %d weights, want %d: %w", i, j, len(weights[i][j]), len(neuron.In), ErrDimension)
			}
		}
	}
	return nil
}

// Weights returns all weights in sequence
func (n *Neural) Weights() [][][]float64 {
	weights := make([][][]float64, len(n.Layers))
	for i, l := range n.Layers {
		weights[i] = make([][]float64, len(l.Neurons))
		for j, n := range l.Neurons {
			weights[i][j] = make([]float64, len(n.In))
			for k, in := range n.In {
				weights[i][j][k] = in.Weight
			}
		}
	}
	return weights
}

// Dump generates a network dump
func (n *Neural) Dump() *Dump {
	config := *n.Config
	config.Layout = slices.Clone(n.Config.Layout)
	return &Dump{
		Config:  &config,
		Weights: n.Weights(),
	}
}

// FromDump restores a Neural from a dump, including dumps written by earlier versions
func FromDump(dump *Dump) (*Neural, error) {
	if dump == nil || dump.Config == nil {
		return nil, fmt.Errorf("dump has no config: %w", ErrInvalidConfig)
	}
	config := *dump.Config
	config.Layout = slices.Clone(dump.Config.Layout)
	config.Weight = zeroWeight
	migrate(&config)

	// before allocating, check that the dump really holds that many weights
	if err := checkDumpSize(&config, dump.Weights); err != nil {
		return nil, fmt.Errorf("restoring: %w", err)
	}
	n, err := NewNeural(&config)
	if err != nil {
		return nil, fmt.Errorf("restoring: %w", err)
	}
	if err := n.ApplyWeights(withOutputBiases(n, dump.Weights)); err != nil {
		return nil, fmt.Errorf("restoring: %w", err)
	}
	return n, nil
}

// migrate rewrites configs that earlier versions accepted into equivalent valid ones
func migrate(c *Config) {
	// multi-label used to default to cross entropy while training on binary cross entropy
	if c.Mode == ModeMultiLabel && c.Loss == LossCrossEntropy {
		c.Loss = LossBinaryCrossEntropy
	}
	// hidden softmax layers used to pass on their inputs unchanged
	if c.Activation == ActivationSoftmax && len(c.Layout) > 1 {
		c.Activation = ActivationLinear
		if c.Mode == ModeDefault {
			c.Mode = ModeMultiClass // whose output layer is softmax, as it was
		}
	}
	if c.Activation == ActivationSoftmax && c.Mode != ModeDefault {
		c.Activation = ActivationLinear
	}
}

func checkDumpSize(c *Config, weights [][][]float64) error {
	if len(weights) != len(c.Layout) {
		return fmt.Errorf("got weights for %d layers, want %d: %w", len(weights), len(c.Layout), ErrDimension)
	}
	fanIn := c.Inputs
	for i, size := range c.Layout {
		if len(weights[i]) != size {
			return fmt.Errorf("layer %d: got %d neurons, want %d: %w", i, len(weights[i]), size, ErrDimension)
		}
		for j, row := range weights[i] {
			if len(row) < fanIn {
				return fmt.Errorf("layer %d, neuron %d: got %d weights, want at least %d: %w", i, j, len(row), fanIn, ErrDimension)
			}
		}
		fanIn = size
	}
	return nil
}

// withOutputBiases adds zero biases where earlier versions left them out, in regression's output layer
func withOutputBiases(n *Neural, weights [][][]float64) [][][]float64 {
	last := len(n.Layers) - 1
	output := slices.Clone(weights[last])
	for j, neuron := range n.Layers[last].Neurons {
		if len(output[j]) == len(neuron.In)-1 {
			output[j] = append(slices.Clone(output[j]), 0)
		}
	}
	return append(slices.Clone(weights[:last]), output)
}

// Marshal marshals to JSON from network
func (n *Neural) Marshal() ([]byte, error) {
	b, err := json.Marshal(n.Dump())
	if err != nil {
		return nil, fmt.Errorf("marshaling network: %w", err)
	}
	return b, nil
}

// Unmarshal restores network from a JSON blob
func Unmarshal(bytes []byte) (*Neural, error) {
	var dump Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, fmt.Errorf("unmarshaling network: %w", err)
	}
	return FromDump(&dump)
}
