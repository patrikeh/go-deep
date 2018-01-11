package deep

import (
	"math"
)

func GetLoss(loss LossType) Loss {
	switch loss {
	case LossCrossEntropy:
		return CrossEntropy{}
	case LossMeanSquared:
		return MeanSquared{}
	}
	return CrossEntropy{}
}

type LossType int

func (l LossType) String() string {
	switch l {
	case LossCrossEntropy:
		return "Cross Entropy"
	case LossMeanSquared:
		return "Mean Squared Error"
	}
	return "Cross Entropy"
}

const (
	LossNone         LossType = 0
	LossCrossEntropy LossType = 1
	LossMeanSquared  LossType = 2
)

type Loss interface {
	F(estimate, ideal [][]float64) float64
	Df(estimate, ideal, activation float64) float64
}

type CrossEntropy struct{}

func (l CrossEntropy) F(estimate, ideal [][]float64) float64 {
	epsilon := 1e-18

	var sum float64
	for i := range estimate {
		ce := 0.0
		for j := range estimate[i] {
			ce += ideal[i][j] * math.Log(estimate[i][j]+epsilon)
		}

		sum -= ce
	}
	return sum / float64(len(estimate))
}

func (l CrossEntropy) Df(estimate, ideal, activation float64) float64 {
	return estimate - ideal
}

type MeanSquared struct{}

func (l MeanSquared) F(estimate, ideal [][]float64) float64 {
	var sum float64
	for i := 0; i < len(estimate); i++ {
		for j := 0; j < len(estimate[i]); j++ {
			sum += math.Pow(estimate[i][j]-ideal[i][j], 2)
		}
	}
	return sum / float64(len(estimate)*len(estimate[0]))
}

func (l MeanSquared) Df(estimate, ideal, activation float64) float64 {
	return activation * (estimate - ideal)
}
