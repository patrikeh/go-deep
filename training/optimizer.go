package training

import "math"

type Optimizer interface {
	Update(value, gradient, moment float64, idx int) float64
}

type SGD struct {
	lr       float64
	momentum float64
	nesterov bool
}

func NewSGD(lr, momentum float64, nesterov bool) *SGD {
	return &SGD{
		lr:       fparam(lr, 0.01),
		momentum: momentum,
		nesterov: nesterov,
	}
}

func (o *SGD) Update(value, gradient, moment float64, idx int) float64 {
	update := o.momentum*moment - o.lr*gradient

	if o.nesterov {
		return o.momentum*update - o.lr*gradient
	}

	return update
}

type Adam struct {
	lr      float64
	beta    float64
	beta2   float64
	epsilon float64

	v, m map[int]float64
	t    float64
}

func NewAdam(lr, beta, beta2, epsilon float64) *Adam {
	return &Adam{
		lr:      fparam(lr, 0.001),
		beta:    fparam(beta, 0.9),
		beta2:   fparam(beta2, 0.999),
		epsilon: fparam(epsilon, 1e-8),
		v:       make(map[int]float64),
		m:       make(map[int]float64),
	}
}

func (o *Adam) Update(value, gradient, moment float64, idx int) float64 {
	o.t++
	lr_t := o.lr * (math.Sqrt(1.0 - math.Pow(o.beta2, o.t))) /
		(1.0 - math.Pow(o.beta, o.t))
	o.m[idx] = o.beta*o.m[idx] + (1.0-o.beta)*gradient
	o.v[idx] = o.beta2*o.v[idx] + (1.0-o.beta2)*math.Pow(gradient, 2.0)

	return -lr_t * (o.m[idx] / (math.Sqrt(o.v[idx]) + o.epsilon))
}

func fparam(val, fallback float64) float64 {
	if val == 0.0 {
		return fallback
	}
	return val
}

func iparam(val, fallback int) int {
	if val == 0 {
		return fallback
	}
	return val
}
