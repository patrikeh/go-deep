package deep

import (
	"encoding/json"
)

type Dump struct {
	Config     *Config
	Weights    [][][]float64
	OutWeights []float64
}

func (n Neural) Dump() *Dump {
	weights := make([][][]float64, len(n.Layers))
	for i, l := range n.Layers {
		weights[i] = make([][]float64, len(l))
		for j, n := range l {
			weights[i][j] = make([]float64, len(n.In))
			for k, in := range n.In {
				weights[i][j][k] = in.Weight
			}
		}
	}
	outWeights := make([]float64, len(n.Layers[len(n.Layers)-1]))
	for i, n := range n.Layers[len(n.Layers)-1] {
		outWeights[i] = n.Out[0].Weight
	}
	return &Dump{
		Config:     n.Config,
		Weights:    weights,
		OutWeights: outWeights,
	}
}

func FromDump(dump *Dump) *Neural {
	neural := NewNeural(dump.Config)

	for i, l := range neural.Layers {
		for j, n := range l {
			for k := range n.In {
				neural.Layers[i][j].In[k].Weight = dump.Weights[i][j][k]
			}
			act := GetActivation(neural.Config.Activation)
			if i == len(neural.Layers)-1 && neural.Config.OutActivation != ActivationNone {
				act = GetActivation(neural.Config.OutActivation)
			}
			n.Activation = act
		}
	}
	for i, n := range neural.Layers[len(neural.Layers)-1] {
		n.Out[0].Weight = dump.OutWeights[i]
	}

	return neural
}

func (n Neural) Marshal() ([]byte, error) {
	d := n.Dump()
	return json.Marshal(d)
}

func Unmarshal(bytes []byte) (*Neural, error) {
	var dump Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return FromDump(&dump), nil
}
