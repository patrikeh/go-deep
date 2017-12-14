package deep

import "math/rand"

type WeightInitializer func() float64

func NewUniform(stdDev, mean float64) WeightInitializer {
	return func() float64 { return Uniform(stdDev, mean) }
}
func Uniform(stdDev, mean float64) float64 {
	return (2*(rand.Float64()-0.5))*stdDev + mean

}

func NewNormal(stdDev, mean float64) WeightInitializer {
	return func() float64 { return Normal(stdDev, mean) }
}
func Normal(stdDev, mean float64) float64 {
	return rand.NormFloat64()*stdDev + mean
}
