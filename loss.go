package deep

import (
	"math"
)

// GetLoss returns a loss function given a LossType
func GetLoss(loss LossType) Loss {
	switch loss {
	case LossCrossEntropy:
		return CrossEntropy{}
	case LossMeanSquared:
		return MeanSquared{}
	case LossBinaryCrossEntropy:
		return BinaryCrossEntropy{}
	}
	return CrossEntropy{}
}

// LossType represents a loss function
type LossType int

func (l LossType) String() string {
	switch l {
	case LossCrossEntropy:
		return "CE"
	case LossBinaryCrossEntropy:
		return "BinCE"
	case LossMeanSquared:
		return "MSE"
	}
	return "N/A"
}

const (
	// LossNone signifies unspecified loss
	LossNone LossType = 0
	// LossCrossEntropy is cross entropy loss
	LossCrossEntropy LossType = 1
	// LossBinaryCrossEntropy is the special case of binary cross entropy loss
	LossBinaryCrossEntropy LossType = 2
	// LossMeanSquared is MSE
	LossMeanSquared LossType = 3
)

// Loss is satisfied by loss functions
type Loss interface {
	F(estimate, ideal [][]float64) float64
	Df(estimate, ideal, activation float64) float64
}

// CrossEntropy is CE loss
type CrossEntropy struct{}

// F is CE(...)
func (l CrossEntropy) F(estimate, ideal [][]float64) float64 {

	var sum float64
	for i := range estimate {
		ce := 0.0
		for j := range estimate[i] {
			ce += ideal[i][j] * math.Log(estimate[i][j])
		}

		sum -= ce
	}
	return sum / float64(len(estimate))
}

// Df is CE'(...)
func (l CrossEntropy) Df(estimate, ideal, activation float64) float64 {
	return estimate - ideal
}

// BinaryCrossEntropy is binary CE loss
type BinaryCrossEntropy struct{}

// F is CE(...)
func (l BinaryCrossEntropy) F(estimate, ideal [][]float64) float64 {
	epsilon := 1e-16
	var sum float64
	for i := range estimate {
		ce := 0.0
		for j := range estimate[i] {
			ce += ideal[i][j]*math.Log(estimate[i][j]+epsilon) + (1.0-ideal[i][j])*math.Log(1.0-estimate[i][j]+epsilon)
		}
		sum -= ce
	}
	return sum / float64(len(estimate))
}

// Df is CE'(...)
func (l BinaryCrossEntropy) Df(estimate, ideal, activation float64) float64 {
	return estimate - ideal
}

// MeanSquared in MSE loss
type MeanSquared struct{}

// F is MSE(...)
func (l MeanSquared) F(estimate, ideal [][]float64) float64 {
	var sum float64
	for i := 0; i < len(estimate); i++ {
		for j := 0; j < len(estimate[i]); j++ {
			sum += math.Pow(estimate[i][j]-ideal[i][j], 2)
		}
	}
	return sum / float64(len(estimate)*len(estimate[0]))
}

// Df is MSE'(...)
func (l MeanSquared) Df(estimate, ideal, activation float64) float64 {
	return activation * (estimate - ideal)
}
