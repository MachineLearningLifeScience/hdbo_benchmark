# Benchmark on toy objective

This experiment benchmarks several baselines and methods for high-dimensional Bayesian Optimization on a simple toy objective: ...

## How to run

```bash
# From the hdbo environment
python src/hdbo_benchmark/experiments/benchmark_on_toy_objective/run.py \
    --function-name=hartmann_6d \
    --solver-name=vanilla_bo \
    --initial-sample-size=10 \
    --n-dimensions=6 \
    --max-iter=300 \
    --seed=1
```

## Results and reporting

Check weights and biases.
