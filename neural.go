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
	Loss       LossType
	Bias       bool
}

func NewNeural(c *Config) *Neural {

	if c.Weight == nil {
		mean := 0.0
		if c.Activation == ActivationReLU {
			mean = 0.1
		}
		c.Weight = NewUniform(0.5, mean)
	}
	if c.Activation == ActivationNone {
		c.Activation = ActivationSigmoid
	}
	if c.Loss == LossNone {
		switch c.Mode {
		case ModeMulticlass, ModeMultiLabel:
			c.Loss = LossCrossEntropy
		case ModeBinary:
			c.Loss = LossBinaryCrossEntropy
		default:
			c.Loss = LossMeanSquared
		}
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

	var biases [][]*Synapse
	if c.Bias {
		biases = make([][]*Synapse, len(layers))
		for i := 0; i < len(layers); i++ {
			if c.Mode == ModeRegression && i == len(layers)-1 {
				continue
			}
			biases[i] = layers[i].ApplyBias(c.Weight)
		}
	}

	return &Neural{
		Layers: layers,
		Biases: biases,
		Config: c,
	}
}

func (n *Neural) Fire() {
	for i := range n.Biases {
		for j := range n.Biases[i] {
			n.Biases[i][j].Fire(1)
		}
	}
	for _, l := range n.Layers {
		l.Fire()
	}
}

func (n *Neural) Forward(input []float64) {
	if len(input) != n.Config.Inputs {
		glog.Errorf("Invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	for _, n := range n.Layers[0].Neurons {
		for i := 0; i < len(input); i++ {
			n.In[i].Fire(input[i])
		}
	}
	n.Fire()
}

func (n *Neural) Predict(input []float64) []float64 {
	n.Forward(input)

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
