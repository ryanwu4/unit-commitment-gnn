using HiGHS
using UnitCommitment
using Dates
using JSON
using JuMP

# Create directory for solutions
solutions_dir = "case118_solutions"
if !isdir(solutions_dir)
    mkdir(solutions_dir)
end

# Get all available dates from the case118_data directory
data_dir = "case118_data"
json_files = filter(f -> endswith(f, ".json"), readdir(data_dir))
dates = sort([replace(f, ".json" => "") for f in json_files])

# Start from index
START_INDEX = 363
dates = dates[START_INDEX:end]

println("Found $(length(dates)) benchmark cases to solve (starting from index $START_INDEX)")
println("Starting optimization runs...")
println("=" ^ 80)

# Track statistics
success_count = 0
failure_count = 0
failed_cases = []

# Process each date
for (idx, date_str) in enumerate(dates)
    global success_count, failure_count, failed_cases
    try
        println("\n[$(idx + START_INDEX - 1)/$(length(dates) + START_INDEX - 1)] Processing: $date_str")
        
        # Read the benchmark instance
        instance = UnitCommitment.read_benchmark("matpower/case118/$date_str")
        
        # Build and solve the model
        model = UnitCommitment.build_model(
            instance = instance, 
            optimizer = HiGHS.Optimizer
        )
        
        # Optimize
        UnitCommitment.optimize!(model)
        
        # Get solution
        solution = UnitCommitment.solution(model)
        
        # Save solution to file
        output_file = joinpath(solutions_dir, "$(date_str)_solution.json")
        UnitCommitment.write(output_file, solution)
        
        println("  ✓ Solution saved to: $output_file")
        
        success_count += 1
        
    catch e
        println("  ✗ ERROR: Failed to solve $date_str")
        println("  ✗ Error message: $e")
        failure_count += 1
        push!(failed_cases, date_str)
    end
    
    # Print progress every 50 cases
    if idx % 50 == 0
        println("\n" * "=" ^ 80)
        println("Progress: $(idx + START_INDEX - 1)/$(length(dates) + START_INDEX - 1) completed")
        println("Success: $success_count | Failures: $failure_count")
        println("=" ^ 80)
    end
end

# Final summary
println("\n" * "=" ^ 80)
println("FINAL SUMMARY")
println("=" ^ 80)
println("Total cases processed: $(length(dates)) (started from index $START_INDEX)")
println("Successful: $success_count")
println("Failed: $failure_count")
println("Solutions saved to: $solutions_dir/")

if !isempty(failed_cases)
    println("\nFailed cases:")
    for case in failed_cases
        println("  - $case")
    end
end

println("\nAll done! ✓")
