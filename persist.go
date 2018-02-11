package deep

import (
	"encoding/json"
)

// Dump is a neural network dump
type Dump struct {
	Config  *Config
	Weights [][][]float64
}

// ApplyWeights sets the weights from a three-dimensional slice
func (n *Neural) ApplyWeights(weights [][][]float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				n.Layers[i].Neurons[j].In[k].Weight = weights[i][j][k]
			}
		}
	}
}

// Weights returns all weights in sequence
func (n Neural) Weights() [][][]float64 {
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
func (n Neural) Dump() *Dump {
	return &Dump{
		Config:  n.Config,
		Weights: n.Weights(),
	}
}

// FromDump restores a Neural from a dump
func FromDump(dump *Dump) *Neural {
	n := NewNeural(dump.Config)
	n.ApplyWeights(dump.Weights)

	return n
}

// Marshal marshals to JSON from network
func (n Neural) Marshal() ([]byte, error) {
	return json.Marshal(n.Dump())
}

// Unmarshal restores network from a JSON blob
func Unmarshal(bytes []byte) (*Neural, error) {
	var dump Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return FromDump(&dump), nil
}
