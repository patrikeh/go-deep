package deep

import (
	"github.com/golang/glog"
)

type Neural struct {
	Layers []Layer
	Biases [][]*Synapse
	Config *Config
}

type Config struct {
	Inputs     int
	Layout     []int
	Activation Activation
	Weight     WeightInitializer
	Bias       float64
}

func NewNeural(c *Config) *Neural {

	layers := make([]Layer, len(c.Layout))
	for i := range layers {
		layers[i] = NewLayer(c.Layout[i], c.Activation)
	}

	// Connect layers & apply biases
	biases := make([][]*Synapse, len(layers)-1)
	for i := 0; i < len(layers)-1; i++ {
		biases[i] = make([]*Synapse, len(layers[i+1]))
		layers[i+1].ApplyBias(biases[i], c.Weight)
		layers[i].Connect(layers[i+1], c.Weight)
	}

	// Set in synapses
	for _, neuron := range layers[0] {
		neuron.In = make([]*Synapse, c.Inputs)
		for i := range neuron.In {
			neuron.In[i] = NewSynapse(c.Weight())
		}
	}

	// Set out synapses
	for _, neuron := range layers[len(layers)-1] {
		neuron.Out = []*Synapse{NewSynapse(c.Weight())}
	}

	return &Neural{
		Layers: layers,
		Biases: biases,
		Config: c,
	}
}

func (n *Neural) Fire() {
	for _, l := range n.Layers {
		l.Fire()
	}
	for _, biases := range n.Biases {
		for _, bias := range biases {
			bias.Fire(n.Config.Bias)
		}
	}
}

func (n *Neural) set(input []float64) {
	for _, n := range n.Layers[0] {
		if len(n.In) != len(input) {
			glog.Errorf("Invalid input dimension - expected: %d got: %d", len(n.In), len(input))
		}
		for i, s := range n.In {
			s.Fire(input[i])
		}
	}
}
func (n *Neural) Feed(input []float64) []float64 {
	n.set(input)
	n.Fire()

	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]float64, len(outLayer))
	for i, neuron := range outLayer {
		out[i] = neuron.Value
	}
	return out
}
