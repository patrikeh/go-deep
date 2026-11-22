module github.com/patrikeh/go-deep

go 1.26

require github.com/stretchr/testify v1.12.1

require go.yaml.in/yaml/v3 v3.0.5 // indirect

// The tests rely on rand.Seed, which is a no-op by default from Go 1.24.
godebug randseednop=0
