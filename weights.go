package deep

import "math/rand"

type WeightInitializer func() float64

func Random() float64 {
	return rand.Float64() - 0.5
}

func NewNormal(stdDev, mean float64) WeightInitializer {
	return func() float64 { return Normal(stdDev, mean) }
}
func Normal(stdDev, mean float64) float64 {
	return rand.NormFloat64()*stdDev + mean
}
