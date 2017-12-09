package deep

import "math"

type ErrorMeasure func(estimate, actual [][]float64) float64

func MSE(estimate, actual [][]float64) float64 {

	var sum float64
	for i := 0; i < len(estimate); i++ {
		for j := 0; j < len(estimate[i]); j++ {
			sum += math.Pow(estimate[i][j]-actual[i][j], 2)
		}
	}
	return sum / float64(len(estimate)*len(estimate[0]))
}
