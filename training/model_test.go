package training

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_SplitSize(t *testing.T) {
	e := make(Examples, 10)

	batches := e.SplitSize(2)
	assert.Len(t, batches, 5)
	for _, batch := range batches {
		assert.Equal(t, 2, len(batch))
	}
}

func Test_SplitN(t *testing.T) {
	e := make(Examples, 10)

	partitions := e.SplitN(3)
	assert.Len(t, partitions, 3)
	assert.Len(t, partitions[0], 4)
	assert.Len(t, partitions[1], 3)
	assert.Len(t, partitions[2], 3)
}

func Test_Split(t *testing.T) {
	rand.Seed(0)

	e := make(Examples, 100)

	a, b := e.Split(0.5)

	assert.InEpsilon(t, len(a), 50, 0.1)
	assert.InEpsilon(t, len(b), 50, 0.1)
}
