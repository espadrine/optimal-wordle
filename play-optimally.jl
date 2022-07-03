using Debugger
using Printf
using Test
import Base.string

const ANSI_RESET_LINE = "\x1b[1K\x1b[G"

function main()
  words = map(s -> Vector{UInt8}(s), readlines("solutions"))
  non_solution_words = map(s -> Vector{UInt8}(s), readlines("non-solution-guesses"))
  allowed_guesses = vcat(words, non_solution_words)

  remaining_solutions = copy(words)  # List of words that currently fit all known constraints.
  tree = newTree(allowed_guesses, remaining_solutions, nothing, UInt8(0))
  while length(remaining_solutions) > 1
    step = 0
    if length(remaining_solutions) == 2315
      @time while tree.best_choice.best_lower_bound < -3.4212
        add_time(computation_timers.improve, @elapsed improve!(tree, remaining_solutions, allowed_guesses))
        step += 1

        choice = tree.best_choice
        println("Times: ", string(computation_timers))
        print("We suggest ", str_from_word(choice.guess), " (",
                @sprintf("%.4f", -choice.best_lower_bound), "~",
                @sprintf("%.4f", -choice.measurement.asymptote),
                "[mse=", @sprintf("%.5f", salet_accuracy), "];",
                @sprintf("s=%d%%", round(choice.prob_beat_best * 100)), ")",
                " step ", step, ". ")
      end
    else
      @time for _ in 1:1000
        improve!(tree, remaining_solutions, allowed_guesses)
        step += 1

        choice = tree.best_choice
        print(ANSI_RESET_LINE)
        print("We suggest ", str_from_word(choice.guess), " (",
                @sprintf("%.4f", -choice.best_lower_bound), "~",
                @sprintf("%.4f", -choice.measurement.asymptote), ";",
                @sprintf("s=%d%%", round(choice.prob_beat_best * 100)), ")",
                " step ", step, ". ")
      end
    end
    println()

    println("Insert your guess: ")
    guess = Vector{UInt8}(readline(stdin))
    println("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ")
    constraint_template = readline(stdin)
    constraint = parse_constraints(constraint_template)
    # TODO: build subtree when it does not exist.
    tree = tree.choices[findfirst(c -> c.guess == guess, tree.choices)].constraints[constraint + 1]
    remaining_solutions = filter_solutions_by_constraint(remaining_solutions, guess, constraint)
    println("Remaining words: ", join(map(s -> str_from_word(s), remaining_solutions), ", "))
  end
  println("Solution: ", str_from_word(remaining_solutions[1]), ".")
end

mutable struct ConvergingMeasurementDifferentials
  visits::Int
  max_visits::Int  # Largest number of visits in ConvergingMeasurements.
  # List of average diffs between each measurement per number of visits.
  slopes::Vector{Float64}
  slope_visits::Vector{Int}
  # Sum of slopes between a given visit count until the latest.
  biases::Vector{Float64}
  biases_visits::Vector{Int}

  # Accuracy computation arises from the difference between the start and
  # converged measurement value.
  init_slope::Float64  # Average slope of the first half of the data points.
  # Parameters for an exponential convergence:
  # estimated_measurement(visits) = asymptote + exp_coeff × exp_base^visits
  exp_base::Float64
  exp_coeff::Float64
  # Precision of the asymptote on a given visits.
  variance::Vector{Float64}
  variance_visits::Vector{Int}
  # y = ax^b, where a = coeff, b = exp, y = variance, x = aggregate visit.
  variance_coeff::Float64
  variance_exp::Float64
end

function newConvergingMeasurementDifferentials()
  visits = 0
  max_visits = visits
  slopes = []
  slope_visits = []
  biases = [0]
  biases_visits = [0]
  init_slope = 0
  exp_base = 0
  exp_coeff = 0
  variance = [0]
  variance_visits = [0]
  variance_coeff = 0
  variance_exp = 0
  return ConvergingMeasurementDifferentials(visits, max_visits, slopes, slope_visits, biases, biases_visits, init_slope, exp_base, exp_coeff, variance, variance_visits, variance_coeff, variance_exp)
end

mutable struct ConvergingMeasurement
  latest::Float64
  visits::Int
  current::Float64  # Smoothed latest value
  average::Float64
  # Estimation of where the sequence converges.
  # Always computed directly from the previous parameters.
  asymptote::Float64
  # List of estimated asymptotes for each exploration.
  asymptotes::Vector{Float64}
  asymptote_mean::Float64
  asymptote_variancet::Float64  # Precision of the asymptote, as variance times N
  current_slope::Float64  # Average slope of the second half.
  differentials::ConvergingMeasurementDifferentials
end

function newConvergingMeasurement(differentials::ConvergingMeasurementDifferentials)::ConvergingMeasurement
  latest = 0
  visits = 0
  current = latest
  average = latest
  asymptote = latest
  asymptotes = []
  asymptote_mean = latest
  asymptote_variancet = 0
  current_slope = 0
  return ConvergingMeasurement(latest, visits, current, average, asymptote, asymptotes, asymptote_mean, asymptote_variancet, current_slope, differentials)
end

function newConvergingMeasurement()::ConvergingMeasurement
  return newConvergingMeasurement(newConvergingMeasurementDifferentials())
end

# Visits are 0-indexed.
function add_measurement!(aggregate::ConvergingMeasurement, new_measurement::Float64)
  aggregate.visits += 1
  aggregate.differentials.max_visits = max(aggregate.differentials.max_visits, aggregate.visits)
  # Update value statistics.
  old_measurement = aggregate.latest
  aggregate.latest = new_measurement
  aggregate.average = (aggregate.average * (aggregate.visits-1) + new_measurement) / aggregate.visits
  aggregate.current = if aggregate.visits < 2
    new_measurement
  else
    (aggregate.current + new_measurement) / 2
  end

  update_differentials!(aggregate, new_measurement - old_measurement)
  #update_exponential_params!(aggregate)
  add_asymptote!(aggregate)
end

function update_differentials!(aggregate::ConvergingMeasurement, measured_diff::Float64)
  if aggregate.visits < 2  # Need 2 points for a diff.
    return
  end
  differentials = aggregate.differentials

  # Update tree-wide slopes.
  slope_visits = aggregate.visits - 1
  if slope_visits > length(differentials.slopes)
    push!(differentials.slopes, 0)
    push!(differentials.slope_visits, 0)
    push!(differentials.biases, 0)
    push!(differentials.biases_visits, 0)
  end
  differentials.slope_visits[slope_visits] += 1
  old_slope = differentials.slopes[slope_visits]
  new_slope = streamed_mean(old_slope, measured_diff, differentials.slope_visits[slope_visits])
  differentials.slopes[slope_visits] = new_slope

  # Update biases.
  nudge = new_slope - old_slope
  for i = 1:aggregate.visits-1
    differentials.biases_visits[i] += 1
    differentials.biases[i] += nudge
  end

  # Exponential weighing of (½^i)÷2 to smooth the differential.
  # That way, the last weighs ½; the nearest 5th measurement still weighs >1%.
  aggregate.current_slope = if aggregate.visits == 2
    measured_diff
  else
    (aggregate.current_slope + measured_diff) / 2
  end

  # The initial slope is the average of the first diff in differentials.
  if aggregate.visits == 2
    differentials.visits += 1
    differentials.init_slope = (differentials.init_slope * (differentials.visits-1) + measured_diff) / differentials.visits
  end
end

# Average the exponential base and coefficient.
#function update_exponential_params!(aggregate::ConvergingMeasurement)
#  if aggregate.visits < 3
#    return  # We need at least 3 points.
#  end
#  differentials = aggregate.differentials
#
#  # Slopes.
#  slope_0 = 0
#  slope_1 = 0
#  slope_count = length(differentials.slopes)
#  slope_mid = slope_count ÷ 2
#  for diff in differentials.slopes[1:slope_mid]
#    slope_0 += diff
#  end
#  for diff in differentials.slopes[slope_mid+1:slope_count]
#    slope_1 += diff
#  end
#
#  # yi = a + bc^i
#  # => c = (y'i ÷ y'0)^(1/i))
#  # => c = ((y[n]-y[n/2]) ÷ (y[n/2]-y[0]))^(2/n))
#  new_exp_base = if slope_0 == 0
#    0.0
#  else
#    diff_quotient = slope_1 / slope_0
#    if diff_quotient < 0
#      0.0
#    else
#      diff_quotient^(2/slope_count)
#    end
#  end
#  differentials.exp_base = new_exp_base
#  if differentials.exp_base <= 0 || differentials.exp_base >= 1
#    return
#  end
#
#  # => b = y'1 ÷ (c×log(c))
#  new_exp_coeff = differentials.init_slope / (differentials.exp_base * log(differentials.exp_base))
#  differentials.exp_coeff = new_exp_coeff
#end

# Add asymptote estimate for the current number of visits.
# Only trigger this when adding a new measurement,
# and after having incremented aggregate.visits.
function add_asymptote!(aggregate::ConvergingMeasurement)
  differentials = aggregate.differentials
  new_asymptote_value = update_asymptote!(aggregate)
  push!(aggregate.asymptotes, new_asymptote_value)

  # Update asymptote mean and variance.
  old_asymptote_mean = aggregate.asymptote_mean
  old_asymptote_variancet = aggregate.asymptote_variancet
  new_asymptote_count = aggregate.visits
  if new_asymptote_count != length(aggregate.asymptotes)
    println("Invalid new_asymptote_count: ", new_asymptote_count, " while the registered asymptotes are ", aggregate.asymptotes)
  end
  new_asymptote_mean = streamed_mean(old_asymptote_mean, new_asymptote_value, new_asymptote_count)
  aggregate.asymptote_mean = new_asymptote_mean
  aggregate.asymptote_variancet = streamed_variance_times_count(
    old_asymptote_variancet, old_asymptote_mean, new_asymptote_mean, new_asymptote_value)

  # If we are the first to register a variance for this visit count, add it.
  if length(differentials.variance) < aggregate.visits
    push!(differentials.variance, 0)
    push!(differentials.variance_visits, 0)
  end
  new_variance_value = aggregate.asymptote_variancet / (new_asymptote_count - 1)
  if !isnan(new_variance_value)
    new_var_visits = differentials.variance_visits[aggregate.visits] + 1
    differentials.variance_visits[aggregate.visits] = new_var_visits
    # We include the updated variance into the tree variance average.
    old_variance = differentials.variance[aggregate.visits]
    differentials.variance[aggregate.visits] = streamed_mean(old_variance, new_variance_value, new_var_visits)
  end
end

function update_asymptote!(aggregate::ConvergingMeasurement)::Float64
  aggregate.asymptote = estimate_asymptote(aggregate)
  return aggregate.asymptote
end

function estimate_asymptote(aggregate::ConvergingMeasurement)::Float64
  return estimate_asymptote_from_bias(aggregate)
end

# Estimate the measured asymptote from exponential regression.
function estimate_asymptote_from_exp_reg(aggregate::ConvergingMeasurement)::Float64
  differentials = aggregate.differentials
  # yi = a + bc^i => a = yi - bc^i
  return aggregate.current - differentials.exp_coeff*differentials.exp_base^aggregate.visits
end

function estimate_bias(aggregate::ConvergingMeasurement)::Float64
  differentials = aggregate.differentials
  bias = 0
  v = max(aggregate.visits, 1)
  for i = v:differentials.max_visits-1
    bias += differentials.slopes[i]
  end
  return bias
end

# Compute the bias that measurements incur by summing the averaged diffs between
# each sequential measurement.
function estimate_asymptote_from_bias(aggregate::ConvergingMeasurement)::Float64
  return aggregate.latest + aggregate.differentials.biases[aggregate.visits]
  #differentials = aggregate.differentials
  #bias = 0
  #v = max(aggregate.visits, 1)
  #for i = v:differentials.max_visits-1
  #  bias += differentials.slopes[i]
  #end
  #return aggregate.latest + bias
end

function estimate_variance(aggregate::ConvergingMeasurement)::Float64
  return aggregate.differentials.variance[aggregate.visits]
end

function asymptote_variance(aggregate::ConvergingMeasurement)::Float64
  return estimate_variance(aggregate)
end

function string(aggregate::ConvergingMeasurement)::String
  differentials = aggregate.differentials
  return string("latest=", @sprintf("%.4f", aggregate.latest),
    " v=", @sprintf("%d", aggregate.visits),
    " cur=", @sprintf("%.4f", aggregate.current),
    " avg=", @sprintf("%.4f", aggregate.average),
    " asym=[", join(map(a -> @sprintf("%.4f", a), aggregate.asymptotes[1:min(4, length(aggregate.asymptotes))]), " "), "]",
    " asym_mean=", @sprintf("%.4f", aggregate.asymptote_mean),
    " asym_var=", @sprintf("%.4f", aggregate.asymptote_variancet / aggregate.visits),
    " cur_slope=", @sprintf("%.4f", aggregate.current_slope),
    " init_slope=", @sprintf("%.4f", differentials.init_slope),
    " exp_base=", @sprintf("%.4f", differentials.exp_base),
    " exp_coeff=", @sprintf("%.4f", differentials.exp_coeff),
    " maxv=", @sprintf("%d", differentials.max_visits),
    " dv=", @sprintf("%d", differentials.visits),
    " dvar=[", join(map(a -> @sprintf("%.4f", a), differentials.variance[1:min(4, length(differentials.variance))]), " "), "]",
    " dvar_visits=[", join(map(a -> @sprintf("%.4f", a), differentials.variance_visits[1:min(4, length(differentials.variance_visits))]), " "), "]",
    " dvar_coeff=", @sprintf("%.4f", differentials.variance_coeff),
    " dvar_exp=", @sprintf("%.4f", differentials.variance_exp))
end

# Streamed mean and variance of a sequence of values.
# Welford’s online algorithm, as seen in
# Donald Knuth’s Art of Computer Programming, Vol 2, page 232, 3rd edition.

function streamed_mean(old_mean, new_value, new_count)
  return old_mean + (new_value - old_mean) / new_count
end

# The variance is the output divided by the number of samples minus 1,
# in order to avoid numerical errors and biases.
function streamed_variance_times_count(old_variance, old_mean, new_mean, new_value)
  return old_variance + (new_value-old_mean) * (new_value-new_mean)
end

mutable struct Choice
  tree::Any
  guess::Vector{UInt8}

  # Including this guess, how many guesses until the win using the current best
  # policy. It is always a lower bound for the optimal value.
  best_lower_bound::Float64

  # The following probabilities are local estimations;
  # they must be divided by sum_prob_optimal etc. to be coherent.
  prob_optimal::Float64
  prob_improvement::Float64
  prob_beat_best::Float64  # Probability that it can beat the best choice.
  # Constraint that Wordle may yield as a response to this guess.
  # We link it lazily to the next choice we make in the tree search.
  visits::Int
  last_visit::Int  # Number of tree visits during last exploration.
  visits_with_improvement::Int

  measurement::ConvergingMeasurement

  constraints::Union{Vector{Any}, Nothing}
end

mutable struct Tree
  previous_choice::Union{Choice, Nothing}
  constraint::UInt8
  choices::Vector{Choice}
  choice_from_guess::Dict{Vector{UInt8}, Choice}
  best_choice::Union{Choice, Nothing}
  # Best known choice based on the known lower bound explored.
  best_choice_lower_bound::Union{Choice, Nothing}
  nsolutions::Int
  visits::Int
  sum_prob_optimal::Float64
  sum_prob_beat_best::Float64
  prob_uncached_beat_best::Float64
  last_non_cache_visit::Int  # Visit count when we last included a guess in the list of choices.
  differentials::ConvergingMeasurementDifferentials
end

function newChoice(guess::Vector{UInt8})::Choice
  measurement = newConvergingMeasurement()
  optimal_estimate = -1
  add_measurement!(measurement, optimal_estimate)

  tree = nothing
  best_lower_bound = -1
  prob_optimal = 1
  prob_improvement = 0
  prob_beat_best = 0
  visits = 0
  last_visit = -1
  visits_with_improvement = 0
  constraints = nothing
  Choice(tree, guess, best_lower_bound, prob_optimal, prob_improvement, prob_beat_best, visits, last_visit, visits_with_improvement, measurement, constraints)
end

function newChoice(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}}, optimal_estimate::Float64, optimal_estimate_variance::Float64, differentials::ConvergingMeasurementDifferentials)::Choice
  measurement = newConvergingMeasurement(differentials)
  #measurement.variancet = optimal_estimate_variance
  add_measurement!(measurement, optimal_estimate)

  tree = nothing
  nsols = Float64(length(solutions))
  prob_optimal = 1
  prob_improvement = 1
  prob_beat_best = 1
  visits = 0
  last_visit = -1
  visits_with_improvement = 0
  constraints = nothing
  Choice(tree, guess, -nsols, prob_optimal, prob_improvement, prob_beat_best, visits, last_visit, visits_with_improvement, measurement, constraints)
end

function newTree(guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}}, previous_choice::Union{Choice, Nothing}, constraint::UInt8)::Tree
  nsols = length(solutions)
  if nsols == 1
    choice = newChoice(solutions[1])
    best_choice = choice
    best_choice_lower_bound = choice
    visits = 0
    sum_prob_optimal = choice.prob_optimal
    sum_prob_beat_best = choice.prob_beat_best
    prob_uncached_beat_best = 0
    last_non_cache_visit = -1
    tree = Tree(previous_choice, constraint, [choice], Dict([(choice.guess, choice)]), best_choice, best_choice_lower_bound, nsols, visits, sum_prob_optimal, sum_prob_beat_best, prob_uncached_beat_best, last_non_cache_visit, choice.measurement.differentials)
    choice.tree = tree
    return tree
  end

  visits = 0
  sum_prob_optimal = 0
  sum_prob_beat_best = 0
  prob_uncached_beat_best = 1
  last_non_cache_visit = -1
  tree = Tree(previous_choice, constraint, [], Dict{Vector{UInt8}, Choice}(), nothing, nothing, nsols, visits, sum_prob_optimal, sum_prob_beat_best, prob_uncached_beat_best, last_non_cache_visit, newConvergingMeasurementDifferentials())
  tree.best_choice = add_choice_from_best_uncached_action!(tree, guesses, solutions)
  tree.best_choice_lower_bound = tree.best_choice
  update_prob_explore!(tree)
  return tree
end

# We measure the mean squared error of the optimal estimate for the guess salet.
salet_accuracy = 0
salet_accuracy_count = 0

mutable struct ComputationTimer
  avg_time::Float64
  count::Int
end

function add_time(timer::ComputationTimer, new_time::Float64)
  timer.count += 1
  timer.avg_time = streamed_mean(timer.avg_time, new_time, timer.count)
end

struct ComputationTimers
  improve::ComputationTimer
  select_choice::ComputationTimer
  new_tree::ComputationTimer
  add_measurement::ComputationTimer
  update_prob_explore::ComputationTimer
end

function string(computation_timers::ComputationTimers)
  return string(
    @sprintf("improve: %.2fs; ", computation_timers.improve.avg_time),
    @sprintf("select_choice: %.2fs; ", computation_timers.select_choice.avg_time),
    @sprintf("new_tree: %.2fs; ", computation_timers.new_tree.avg_time),
    @sprintf("add_measurement: %.2fs; ", computation_timers.add_measurement.avg_time),
    @sprintf("update_prob_explore: %.2fs; ", computation_timers.update_prob_explore.avg_time),
  )
end

computation_timers = ComputationTimers(ComputationTimer(0, 0), ComputationTimer(0, 0), ComputationTimer(0, 0), ComputationTimer(0, 0), ComputationTimer(0, 0))

# Improve the policy by gathering data from using it with all solutions.
# Returns the average number of guesses to win across all solutions.
function improve!(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Float64
  nsolutions = length(solutions)
  if nsolutions == 0
    return 0
  elseif nsolutions == 1
    return 1  # The last guess finds the right solution.
  end

  # Select the next choice based on the optimal-converging policy
  add_time(computation_timers.select_choice, @elapsed begin
    choice = best_exploratory_choice_with_ordering!(tree, solutions, guesses)
  end)
  init_prob_explore = choice.prob_beat_best / tree.sum_prob_beat_best
  init_measurement_latest = choice.measurement.latest
  if nsolutions == 2315
    println("Before exploration: ", string(choice.measurement))
  end
  best_guesses_to_win = 0  # Measured best, to update the best score.
  new_tree_optimal_estimate = 0

  # FIXME: speed improvement: loop through solutions if they are less numerous.
  for constraint in UInt8(0):UInt8(242)
    # State transition.
    remaining_solutions = filter_solutions_by_constraint(solutions, choice.guess, constraint)
    nrsols = length(remaining_solutions)
    if nrsols == 0
      continue
    elseif nrsols == 1
      if constraint == 0xf2  # All characters are valid: we won in 1 guess.
        new_tree_optimal_estimate -= 1
        best_guesses_to_win -= 1
      else                   # The solution is found: we win on the next guess.
        new_tree_optimal_estimate -= 2
        best_guesses_to_win -= 2
      end
      continue
    end

    # Subtree construction
    if isnothing(choice.constraints)
      choice.constraints = Vector{Union{Tree, Nothing}}(nothing, 243)
    end
    subtree = choice.constraints[constraint + 1]

    if isnothing(subtree)        # Initialize the next move.
      add_time(computation_timers.new_tree, @elapsed begin
        subtree = newTree(guesses, remaining_solutions, choice, constraint)
        choice.constraints[constraint + 1] = subtree
      end)
    else
      improve!(subtree, remaining_solutions, guesses)
    end

    ## Full-depth:
    #if isnothing(subtree)        # Initialize the next move.
    #  add_time(computation_timers.new_tree, @elapsed begin
    #    subtree = newTree(guesses, remaining_solutions, choice, constraint)
    #    choice.constraints[constraint + 1] = subtree
    #  end)
    #end
    #improve!(subtree, remaining_solutions, guesses)

    # For each solution, we made one guess, on top of the guesses left to end the game.
    best_guesses_to_win += (subtree.best_choice.best_lower_bound - 1) * nrsols
    # FIXME: we should not use the most optimistic estimate,
    # but the expected estimate,
    # by weighing each asymptote by the probability that it is optimal.
    new_tree_optimal_estimate += (subtree.best_choice.measurement.asymptote - 1) * nrsols
  end

  # Update information about the current best policy.
  new_guesses_remaining = best_guesses_to_win / nsolutions
  new_tree_optimal_estimate /= nsolutions
  add_time(computation_timers.add_measurement, @elapsed begin
    add_measurement!(choice, new_tree_optimal_estimate, new_guesses_remaining)
  end)
  add_time(computation_timers.update_prob_explore, @elapsed begin
    update_prob_explore!(tree)
  end)
  #if nsolutions == 2315
  if !isnothing(findfirst(s -> str_from_word(s) == "vowel", solutions))
    println("First choice optimal prob: ", tree.choices[1].prob_optimal)
    print_tree(tree)
    println("Explored ", choice_breadcrumb(choice), ": ", nsolutions, " sols (",
            @sprintf("%.4f", -choice.best_lower_bound), "~",
            @sprintf("%.4f", -init_measurement_latest), "→",
            @sprintf("%.4f", -choice.measurement.latest), "∞→",
            @sprintf("%.4f", -choice.measurement.asymptote), "±",
            @sprintf("%.4f", sqrt(asymptote_variance(choice.measurement))), ";",
            @sprintf("e=%d%%", round(init_prob_explore * 100)), "→",
            @sprintf("%d%%", round(choice.prob_beat_best / tree.sum_prob_beat_best * 100)), ";",
            @sprintf("o=%d%%", round(choice.prob_optimal / tree.sum_prob_optimal * 100)), ";",
            @sprintf("i=%d%%", round(choice.prob_improvement * 100)), ";",
            " visits ", choice.visits, ")")
    println("After exploration: ", string(choice.measurement))
    println()
    if str_from_word(choice.guess) == "salet" && salet_accuracy_count < 100
      global salet_accuracy = ((salet_accuracy * salet_accuracy_count) + (choice.measurement.asymptote - -3.4212)^2) / (salet_accuracy_count+1)
      global salet_accuracy_count += 1
    end
  end
  return choice.best_lower_bound
end

function best_exploratory_choice_with_ordering!(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Choice
  choice, i = best_exploratory_choice(tree, solutions, guesses)
  choice.last_visit = tree.visits
  if i > 1 && choice.prob_beat_best > tree.choices[i-1].prob_beat_best
    # Constant-time iteration that converges to a sorted array.
    tree.choices[i] = tree.choices[i-1]
    tree.choices[i-1] = choice
  end
  return choice
end

function best_exploratory_choice(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
  # What is the cached choice whose frequency elects it first?
  cached_choice, cached_idx, cached_next_visit = best_cached_exploratory_choice(tree)
  if isnothing(cached_choice)
    selection = best_non_cached_exploratory_choice(tree, guesses, solutions)
    if !isnothing(selection)
      if isnothing(tree.previous_choice)
        println("\nChose because no cached choice was available")
      end
      return selection
    end
  end
  # What about the non-cached choices?
  uncached_exploration_prob = tree.prob_uncached_beat_best / tree.sum_prob_beat_best
  non_cached_visit_freq = 1 / uncached_exploration_prob
  non_cached_next_visit = tree.last_non_cache_visit + non_cached_visit_freq
  if cached_next_visit <= non_cached_next_visit
    if isnothing(tree.previous_choice)
      println("\nChose based on cached next visit: ", @sprintf("%.2f", cached_next_visit))
    end
    return cached_choice, cached_idx
  else
    selection = best_non_cached_exploratory_choice(tree, guesses, solutions)
    if isnothing(selection)
      if isnothing(tree.previous_choice)
        println("\nChose based on cached next visit as there are non uncached choices: ", @sprintf("%.2f", cached_next_visit))
      end
      return cached_choice, cached_idx
    end
    if isnothing(tree.previous_choice)
      println("\nChose based on non-cached next visit: ", @sprintf("%.2f", non_cached_next_visit))
    end
    return selection
  end
end

# Returns the choice, its index in the cache, and the next tree visit at which
# it needed to be explored.
function best_cached_exploratory_choice(tree::Tree)::Tuple{Union{Choice, Nothing}, Int, Float64}
  min_next_visit = Inf
  min_choice = nothing
  min_idx = 0
  for (i, choice) in enumerate(tree.choices)
    # The frequency of visits should match the probability of exploration.
    next_visit = if choice.visits <= 0
      0.0
    else
      prob_visit = choice.prob_beat_best / tree.sum_prob_beat_best
      visit_freq = 1 / prob_visit
      choice.last_visit + visit_freq
    end
    if next_visit < min_next_visit
      min_next_visit = next_visit
      min_choice = choice
      min_idx = i
    end
  end
  return min_choice, min_idx, min_next_visit
end

function best_non_cached_exploratory_choice(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Union{Tuple{Choice, Int}, Nothing}
  # Find the remaining guess with the best estimate.
  choice = add_choice_from_best_uncached_action!(tree, guesses, solutions)
  if isnothing(choice)
    return nothing
  end
  tree.last_non_cache_visit = tree.visits
  return choice, length(tree.choices)
end

# Pick the choice based on fair total expanded work: a choice that should be
# explored at 40% should be explored until 40% of all historical explorations
# is theirs.
function fair_exploratory_choice(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
  for (i, choice) in enumerate(tree.choices)
    prob_visit = choice.prob_beat_best / tree.sum_prob_beat_best
    if choice.visits < tree.visits * prob_visit
      return choice, i
    end
  end
  selection = best_non_cached_exploratory_choice(tree, guesses, solutions)
  if isnothing(selection)
    tree.choices[1], 1
  end
  return selection
end

function add_measurement!(choice::Choice, new_measurement::Float64, new_lower_bound::Float64)
  tree = choice.tree

  old_asymptote = choice.measurement.asymptote
  add_measurement!(choice.measurement, new_measurement)
  if choice.measurement.asymptote > old_asymptote
    choice.visits_with_improvement += 1
  end

  if choice.measurement.asymptote > tree.best_choice.measurement.asymptote
    tree.best_choice = choice
  end

  old_tree_best_lower_bound = tree.best_choice_lower_bound.best_lower_bound
  if new_lower_bound > choice.best_lower_bound
    choice.best_lower_bound = new_lower_bound
  end
  if choice.best_lower_bound > old_tree_best_lower_bound
    println("Improvement found: ", choice_breadcrumb(choice), " ",
            @sprintf("%.4f", old_tree_best_lower_bound), "→",
            @sprintf("%.4f", choice.best_lower_bound),
            " (", tree.nsolutions, " sols)")
    tree.best_choice_lower_bound = choice
  end

  # Recursive constant:
  # tree.best_choice.best_lower_bound matches all recursive best choices.
  # We verify it through assertion testing.
  #if nsolutions == 2315
  #  total = total_guesses_for_all_sols(tree, guesses, solutions)
  #  if abs(tree.best_choice.best_lower_bound - total / nsolutions) > 0.01
  #    println("Total guesses: ", total, "; tree best choice guesses: ", tree.best_choice.best_lower_bound)
  #  end
  #end

  # For exploration, we rely on a combined metric that estimates the lower bound
  # of the number of guesses left before winning, based on our uncertainty.
  choice.visits += 1
  tree.visits += 1
end

# The likelihood that we pick a choice is its worthiness:
# the odds that it is optimal and that its exploration improves its score.
function update_prob_explore!(tree::Tree)
  for c in tree.choices
    update_asymptote!(c.measurement)
    #c.measurement.asymptote = estimate_asymptote(c.measurement)
    #c.measurement.variance = estimate_variance(c.measurement)
  end
  #sum_prob_optimal = 0
  #for c in tree.choices
  #  c.prob_optimal = prob_optimal_choice(c.measurement.asymptote, asymptote_variance(c.measurement), tree)
  #  sum_prob_optimal += c.prob_optimal
  #end
  #tree.sum_prob_optimal = sum_prob_optimal + tree.prob_non_cached_optimal
  sum_prob_beat_best = 0
  for c in tree.choices
    #c.prob_improvement = prob_improvement(c)
    #c.prob_beat_best = (c.prob_optimal / tree.sum_prob_optimal) * c.prob_improvement
    c.prob_beat_best = prob_beat_best(c)
    c.prob_improvement = c.prob_beat_best
    sum_prob_beat_best += c.prob_beat_best
  end
  #tree.sum_prob_beat_best = sum_prob_beat_best + tree.prob_non_cached_optimal / tree.sum_prob_optimal
  tree.sum_prob_beat_best = sum_prob_beat_best + tree.prob_uncached_beat_best
end

# Probability that this choice is optimal under perfect play.
function prob_optimal_choice(optimal_estimate::Float64, optimal_estimate_variance::Float64, tree::Tree)::Float64
  if isinf(optimal_estimate)
    return 0
  end
  prob = 1
  for other in tree.choices
    prob *= prob_superior_choice(optimal_estimate, optimal_estimate_variance, other)
  end
  return prob
end

function prob_superior_choice(optimal_estimate::Float64, optimal_estimate_variance::Float64, other::Choice)::Float64
  diff_variance = optimal_estimate_variance + asymptote_variance(other.measurement)
  if diff_variance == 0
    if optimal_estimate >= other.measurement.asymptote
      return 1
    else
      return 0
    end
  end
  # We now pretend the difference between this choice’s distribution
  # and the best choice’s is logistic.
  choice_mu = optimal_estimate  # We consider it to be the mode of its Gumbel.
  other_mu = other.measurement.asymptote
  return 1 - 1 / (1 + exp(-(0-(choice_mu-other_mu))/(sqrt(3*diff_variance)/pi)))
end

# Alternative estimate of the probability of a choice being optimal.
function prob_choice_reaching_optimal(optimal_estimate::Float64, optimal_estimate_variance::Float64, tree::Tree)::Float64
  # We estimate it by comparing the probability that a choice reaches the
  # current overall optimal estimate, under the categorical distribution.
  optimum = if isnothing(tree.best_choice)
    optimal_estimate
  else
    tree.best_choice.measurement.asymptote
  end
  # We assume that optimal estimates follow a Gumbel distribution,
  # since it is the maximum of a set of current-best-measurements
  # which follow an exponential distribution.
  mode = optimal_estimate
  # FIXME: we are comparing optimality probabilities between choices which had a
  # large number of guesses, improving their variance, and choices that did not.
  # Ideally, they should be compared with a variance that they would have after
  # the same number of visits.
  beta = sqrt(optimal_estimate_variance * 6 / pi^2)
  z = (optimum - mode) / beta
  return exp(-(z + exp(-z))) / beta
end

# Probability that the given choice will surpass the best asymptote.
function prob_beat_best(choice::Choice)::Float64
  return prob_beat_best(choice.measurement.asymptote, asymptote_variance(choice.measurement), choice.tree.best_choice.measurement.asymptote)
end

function prob_beat_best(reward::Float64, variance::Float64, best_reward::Float64)::Float64
  return 1 - gumbel_cdf(reward, variance, best_reward)
end

# Probability that exploring this choice will eventually yield an improvement.
#function prob_improvement(choice::Choice)::Float64
#  if isnothing(choice.constraints)
#    if choice.visits == 0
#      return 1
#    else
#      return choice.visits_with_improvement / choice.visits
#    end
#    # Lacking children information, we compute the probability that the
#    # asymptote is actually the lower bound or lower
#    # (given the asymptote's imprecision), which would mean it no longer improves.
#    return 1 - gumbel_cdf(choice.measurement.asymptote, asymptote_variance(choice.measurement), choice.best_lower_bound)
#  else
#    # Recursive computation: it is the probability that each subtree achieves an
#    # improvement, weighed by the probability that that subtree is explored.
#    # - Constraint subtrees are weighted by the number of solutions, since each
#    #   subtree corresponding to one constraint result, will be explored in
#    #   proportion to the number of solutions that match it.
#    # - Choices are weighed by prob_beat_best, as that is the exploration
#    #   probability.
#    sum_subtree_weights = choice.tree.nsolutions
#    return foldl((p, t) -> p + if isnothing(t)
#      0
#    else
#      subtree_weight = t.nsolutions
#      sum_choice_weights = t.sum_prob_beat_best
#      subtree_prob = (foldl(
#        (p, c) -> p + c.prob_improvement * c.prob_beat_best,
#        t.choices,
#        init=0
#       ) + 1 * t.prob_non_cached_optimal / t.sum_prob_optimal) / sum_choice_weights
#      subtree_weight * subtree_prob
#    end, choice.constraints, init=0) / sum_subtree_weights
#  end
#end

function gumbel_cdf(mode, variance, value)
  if variance == 0
    if value >= mode
      return 1  # The PDF spikes at the mode.
    else
      return 0
    end
  end
  beta = sqrt(variance * 6 / pi^2)
  return exp(-exp(-(value-mode)/beta))
end

function gumbel_pdf(mode, variance, value)
  if variance == 0
    if value == mode
      return 1  # The PDF spikes at the mode.
    else
      return 0
    end
  end
  beta = sqrt(variance * 6 / pi^2)
  z = (value - mode) / beta
  p = exp(-(z + exp(-z))) / beta
  if p > 1
    return 1
  end
  return p
end

struct ActionEstimate
  action::Vector{UInt8}
  cumulative_reward::Float64
end

# Use local estimates to pick the best action from the list of uncached actions.
# Convert it to a choice.
function add_choice_from_best_uncached_action!(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Union{Choice, Nothing}
  action_estimates = uncached_action_estimates(tree, guesses, solutions)
  best_action_estimate, second_best_action_estimate = find_best_action_estimate(action_estimates)
  if isnothing(best_action_estimate)
    return nothing
  end
  # We use the second variance, as the variance with a single measurement is 0.
  uncached_variance = if length(tree.differentials.variance) < 2 || tree.differentials.variance[2] == 0
    best_action_estimate.cumulative_reward^2
  else
    tree.differentials.variance[2]
  end

  if isnothing(second_best_action_estimate)
    tree.prob_uncached_beat_best = 0
  else
    # Take the opportunity to update the prob that an uncached action surpasses
    # the best reward across all actions given perfect play.
    best_reward = if isnothing(tree.best_choice)
      best_action_estimate.cumulative_reward
    else
      tree.best_choice.measurement.asymptote
    end
    tree.prob_uncached_beat_best = prob_beat_best(second_best_action_estimate.cumulative_reward, uncached_variance, best_reward)
  end

  choice = newChoice(best_action_estimate.action, solutions, best_action_estimate.cumulative_reward, uncached_variance, tree.differentials)
  choice.tree = tree
  push!(tree.choices, choice)
  tree.choice_from_guess[best_action_estimate.action] = choice
  return choice
end

# Return a map from guesses to local estimates for the number of guesses to win.
function uncached_action_estimates(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Vector{ActionEstimate}
  action_estimates = Vector{ActionEstimate}()
  for guess in guesses
    if !isnothing(tree) && haskey(tree.choice_from_guess, guess)
      continue
    end
    estimated_cumulative_reward = estimate_guesses_remaining(guess, solutions)
    if isinf(estimated_cumulative_reward)
      continue
    end
    bias = if length(tree.differentials.biases) == 0
      0
    else
      tree.differentials.biases[1]
    end
    debiased_reward = estimated_cumulative_reward + bias
    push!(action_estimates, ActionEstimate(guess, debiased_reward))
  end
  return action_estimates
end

function find_best_action_estimate(action_estimates::Vector{ActionEstimate})::Tuple{Union{ActionEstimate, Nothing}, Union{ActionEstimate, Nothing}}
  if length(action_estimates) == 0
    return nothing, nothing
  end
  max = -Inf
  best = action_estimates[1]
  if length(action_estimates) == 1
    return best, nothing
  end
  second_best = action_estimates[2]
  for estimate in action_estimates
    if estimate.cumulative_reward > max
      max = estimate.cumulative_reward
      second_best = best
      best = estimate
    end
  end
  return best, second_best
end

#function add_estimated_best_guess!(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Choice
#  nguesses = length(guesses)
#  choice_estimates = Vector{Float64}(undef, nguesses)
#  best_guess = guesses[1]
#  best_optimal_estimate = -Inf
#  optimal_estimate_mean = 0
#  for (i, guess) in enumerate(guesses)
#    if !isnothing(tree) && haskey(tree.choice_from_guess, guess)
#      continue
#    end
#    optimal_estimate = estimate_guesses_remaining(guess, solutions)
#    if isinf(optimal_estimate)
#      continue
#    end
#    @inbounds choice_estimates[i] = optimal_estimate
#    optimal_estimate_mean += optimal_estimate
#    if optimal_estimate > best_optimal_estimate
#      best_guess = guess
#      best_optimal_estimate = optimal_estimate
#    end
#  end
#  optimal_estimate_mean /= nguesses
#
#  optimal_estimate_variance = 0
#  for (i, _) in enumerate(guesses)
#    @inbounds optimal_estimate_variance += (choice_estimates[i] - optimal_estimate_mean)^2
#  end
#  optimal_estimate_variance /= (nguesses-1)
#
#  # The variance of the mean of a set of random variables Xi is:
#  # var((ΣXi)÷N) = (Σvar(Xi))÷N² = σ²÷N
#  # Thus the average variance of each estimate is this:
#  choice_optimal_estimate_variance = optimal_estimate_variance * nguesses
#
#  choice = newChoice(best_guess, solutions, best_optimal_estimate, choice_optimal_estimate_variance, tree.differentials)
#  choice.tree = tree
#  push!(tree.choices, choice)
#  tree.choice_from_guess[best_guess] = choice
#
#  # What is the probability that one of the non-cached choices is optimal?
#  prob_non_cached_optimal = 0
#  for (i, _) in enumerate(guesses)
#    @inbounds prob_non_cached_optimal += prob_optimal_choice(choice_estimates[i], choice_optimal_estimate_variance, tree)
#  end
#  tree.prob_non_cached_optimal = prob_non_cached_optimal
#  # Possible approximation:
#  # 1. What is the probability that the first non-cached choice is optimal?
#  #prob_non_cached_choice_is_optimal = prob_optimal_choice(second_best_optimal_estimate, choice_optimal_estimate_variance, tree)
#  # 2. Assuming all non-cached choices have this probability, what is the overall probability?
#  #    A choice being optimal is an exclusive event, so we can add them up.
#  #tree.prob_non_cached_optimal = prob_non_cached_choice_is_optimal * (nguesses-1)
#
#  return choice
#end

function estimate_guesses_remaining(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}})
  avg_remaining = average_remaining_solutions_after_guess(guess, solutions)
  nsols = length(solutions)
  prob_sol = if guess in solutions
    1 / nsols  # If this pick is a winner, there are no more guesses to make.
  else
    0
  end
  # To estimate the number of remaining guesses n to win, we assume that we
  # maintain a constant ratio q of removed solutions after each guess.
  # We have s solutions currently, such that q^(n-1) = s. Thus n = 1 + log(s)÷log(q).
  return -(1 + (prob_sol * 0 + (1-prob_sol) * (log(nsols) / log(nsols / avg_remaining))))
end

# Compute the average remaining solutions for each guess.
function average_remaining_solutions_after_guess(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}})
  counts = zeros(Int, 243)
  for solution in solutions
    @inbounds counts[constraints(guess, solution) + 1] += 1
  end
  sum(abs2, counts) / length(solutions)
end

function print_tree(tree::Tree)
  println("tree.sum_prob_optimal = ", tree.sum_prob_optimal)
  println("tree.sum_prob_beat_best = ", tree.sum_prob_beat_best)
  println("tree.prob_uncached_beat_best = ", tree.prob_uncached_beat_best)
  println("tree.slopes = ", join(map(s -> @sprintf("%.3f", s), tree.differentials.slopes[1:min(10, length(tree.differentials.slopes))]), ", "))
  println("tree.biases = ", join(map(s -> @sprintf("%.3f", s), tree.differentials.biases[1:min(10, length(tree.differentials.biases))]), ", "))
  for c in tree.choices
    println(str_from_word(c.guess), " ", @sprintf("%.4f", -c.best_lower_bound),
      "~", @sprintf("%.4f", -c.measurement.latest),
      "→", @sprintf("%.4f", -c.measurement.asymptote),
      "±", @sprintf("%.4f", sqrt(asymptote_variance(c.measurement))),
      " o=", @sprintf("%.4f", c.prob_optimal / tree.sum_prob_optimal),
      " i=", @sprintf("%.4f", c.prob_improvement),
      " b=", @sprintf("%.4f", c.prob_beat_best / tree.sum_prob_beat_best),
      " v=", c.visits,
     )
  end
end

function choice_breadcrumb(choice::Choice)
  choices = str_from_word(choice.guess)
  while !isnothing(choice.tree.previous_choice)
    choices = @sprintf("%s %s %s",
      str_from_word(choice.tree.previous_choice.guess),
      str_from_constraints(choice.tree.constraint),
      choices)
    choice = choice.tree.previous_choice
  end
  return choices
end

function str_from_word(word::Vector{UInt8})::String
  String(copy(word))
end

function constraints(guess::Vector{UInt8}, actual::Vector{UInt8})::UInt8
  constraints = UInt8(0)
  guess_buf = copy(guess)
  actual_buf = copy(actual)
  # We do exact matches in a separate first pass,
  # because they have priority over mispositioned matches.
  mult = 1
  for i in 1:5
    if guess_buf[i] == actual_buf[i]
      constraints += 2 * mult
      actual_buf[i] = 1  # Remove the letter, but without the cost.
      guess_buf[i] = 2   # Remove the letter, but without conflicting with the line above.
    end
    mult *= 3
  end
  mult = 1
  for i in 1:5
    for j in 1:5
      if guess_buf[i] == actual_buf[j]
        constraints += mult
        # Remove the letter so another identical guess letter
        # will not also match it.
        actual_buf[j] = 1
        break
      end
    end
    mult *= 3
  end
  return constraints
end

function filter_solutions_by_constraint(solutions::Vector{Vector{UInt8}}, guess::Vector{UInt8}, constraint::UInt8)::Vector{Vector{UInt8}}
  filter(w -> match_constraints(w, guess, constraint), solutions)
end

# Code: . = has_no, x = has_but_not_at, o = has_at
function parse_constraints(template::String)::UInt8
  constraints = UInt8(0)
  mult = 1
  for c in template
    constraints += if c == 'o'
      2
    elseif c == 'x'
      1
    else
      0
    end * mult
    mult *= 3
  end
  return constraints
end

template_constraint_chars = ".xo"
function str_from_constraints(constraints::UInt8)::String
  template = ""
  for _ in 1:5
    template = string(template, template_constraint_chars[(constraints % 3) + 1])
    constraints ÷= 3
  end
  return template
end

function match_constraints(word::Vector{UInt8}, guess::Vector{UInt8}, given_constraints::UInt8)::Bool
  return constraints(guess, word) == given_constraints
end

function total_guesses_for_all_sols(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Int
  nguesses = 0
  for solution in solutions
    nguesses += total_guesses_for_sol(tree, solution, guesses, solutions)
  end
  nguesses
end

function total_guesses_for_sol(tree::Tree, solution::Vector{UInt8}, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Int
  if length(solutions) == 1
    return 1
  end
  choice = tree.best_choice
  c = constraints(choice.guess, solution)
  if c == 0xf2  # We found the right solution; game over.
    return 1
  end
  remaining_solutions = filter_solutions_by_constraint(solutions, choice.guess, c)
  if length(remaining_solutions) == 1
    return 2    # We guessed wrong, but the next guess will be right.
  end
  return 1 + total_guesses_for_sol(choice.constraints[c + 1], solution, guesses, remaining_solutions)
end


function test()
  constraint_tests = [
    ["erase", "melee", "x...o"],
    ["erase", "agree", "xxx.o"],
    ["erase", "widen", "x...."],
    ["erase", "early", "oxx.."],
    ["erase", "while", "....o"],
    ["alias", "today", "...o."],
    ["chuck", "chunk", "ooo.o"],
  ]

  for test in constraint_tests
    @test constraints(Vector{UInt8}(test[1]), Vector{UInt8}(test[2])) == parse_constraints(test[3])
  end

  constraint_parsing_tests = ["xo..x", ".....", "oxxx."]
  for test in constraint_parsing_tests
    @test str_from_constraints(parse_constraints(test)) == test
  end
end

#test()
main()
