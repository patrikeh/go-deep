package deep

import (
	"fmt"

	"github.com/golang/glog"
)

type Neural struct {
	Layers []*Layer
	Biases [][]*Synapse
	Config *Config
}

type Config struct {
	Inputs     int
	Layout     []int
	Activation ActivationType
	Mode       Mode
	Weight     WeightInitializer `json:"-"`
	Error      ErrorMeasure      `json:"-"`
	Bias       float64
}

func NewNeural(c *Config) *Neural {

	if c.Weight == nil {
		c.Weight = NewUniform(0.5, 0)
	}
	if c.Activation == ActivationNone {
		c.Activation = ActivationSigmoid
	}

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

	biases := make([][]*Synapse, len(layers))
	for i := 0; i < len(layers); i++ {
		biases[i] = layers[i].ApplyBias(c.Weight)
	}

	return &Neural{
		Layers: layers,
		Biases: biases,
		Config: c,
	}
}

func (n *Neural) Fire() {
	for _, biases := range n.Biases {
		for _, bias := range biases {
			bias.Fire(n.Config.Bias)
		}
	}
	for _, l := range n.Layers {
		l.Fire()
	}
}

func (n *Neural) set(input []float64) {
	for _, n := range n.Layers[0].Neurons {
		if len(n.In)-1 != len(input) {
			glog.Errorf("Invalid input dimension - expected: %d got: %d", len(n.In), len(input))
		}
		for i := 0; i < len(n.In)-1; i++ {
			n.In[i].Fire(input[i])
		}
	}
}
func (n *Neural) Forward(input []float64) []float64 {
	n.set(input)
	n.Fire()

	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]float64, len(outLayer.Neurons))
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Value
	}
	return out
}

func (n *Neural) String() string {
	var s string
	for _, l := range n.Layers {
		s = fmt.Sprintf("%s\n%s", s, l)
	}
	return s
}
