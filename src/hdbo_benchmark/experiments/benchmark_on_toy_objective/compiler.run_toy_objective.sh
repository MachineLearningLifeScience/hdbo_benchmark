# Use: ./compiler.run_toy_objective.sh > run.sh
# And then python src/hdbo_benchmark/experiments/pool.py run.sh n_processes_you_want_to_run (e.g. 4)
# Or ./entry.sh n_processes_you_want_to_run

for solver_name in "random_search" "line_bo" "vanilla_bo" "cma_es" "baxus" "saas_bo"
do
    for n_dimensions in 10 50 100 500 1000
    do
        for function_name in "shifted_sphere" "styblinski_tang" "ackley_function_01" "levy" "rosenbrock"
        do
            for seed in 1 2 3 4 5 6 7 8 9 10
            do
                echo "conda run -n hdbo python src/hdbo_benchmark/experiments/benchmark_on_toy_objective/run.py --function-name=${function_name} --solver-name=${solver_name} --initial-sample-size=10 --n-dimensions=${n_dimensions} --max-iter=300 --seed=${seed}"
            done
        done
    done
done