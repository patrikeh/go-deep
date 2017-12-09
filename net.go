package deep

import (
	"github.com/golang/glog"
)

type Neural struct {
	Layers []Layer
}

func NewNeural(
	inputs int,
	layout []int,
	activation Activation,
	weight WeightInitializer) *Neural {

	layers := make([]Layer, len(layout))
	for i := range layers {
		layers[i] = NewLayer(layout[i], activation)
	}
	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], weight)
	}

	// Set in synapses and apply bias
	for _, neuron := range layers[0] {
		neuron.In = make([]*Synapse, inputs+1)
		for i := range neuron.In {
			neuron.In[i] = NewSynapse(weight())
		}
	}

	// Set out synapses
	for _, neuron := range layers[len(layers)-1] {
		neuron.Out = []*Synapse{NewSynapse(weight())}
	}

	return &Neural{
		Layers: layers,
	}
}

func (n *Neural) Fire() {
	for _, l := range n.Layers {
		l.Fire()
	}
}

func (n *Neural) Feed(input []float64) []float64 {
	for _, n := range n.Layers[0] {
		if len(n.In) != len(input) {
			glog.Errorf("Invalid input dimension - expected: %d got: %d", len(n.In), len(input))
		}
		for i, s := range n.In {
			//s.In = input[i]
			s.Fire(input[i])
		}
	}

	n.Fire()

	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]float64, len(outLayer))
	for i, neuron := range outLayer {
		out[i] = neuron.Value
	}
	return out
}
