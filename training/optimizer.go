package training

type Optimizer interface {
	Update(value, gradient, moment float64) float64
}

type SGD struct {
	lr       float64
	momentum float64
	nesterov bool
}

func NewSGD(lr, momentum float64, nesterov bool) SGD {
	return SGD{
		lr:       fparam(lr, 0.01),
		momentum: momentum,
		nesterov: nesterov,
	}
}

func (o SGD) Update(value, gradient, moment float64) float64 {
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
}

func NewAdam(lr, beta, beta2, epsilon float64) Adam {
	return Adam{
		lr:      fparam(lr, 0.1),
		beta:    fparam(beta, 0.99),
		beta2:   fparam(beta2, 0.999),
		epsilon: fparam(epsilon, 1e-8),
	}
}

func (o Adam) Update(value, gradient, moment float64) float64 {
	return 0.0
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
