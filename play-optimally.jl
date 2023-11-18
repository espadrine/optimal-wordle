using Random, Distributions
using Debugger
using Printf
using Test
import Base.string

const ANSI_RESET_LINE = "\x1b[1K\x1b[G"
Random.seed!(1)

function main()
  words = map(s -> Vector{UInt8}(s), readlines("solutions"))
  non_solution_words = map(s -> Vector{UInt8}(s), readlines("non-solution-guesses"))
  allowed_guesses = vcat(words, non_solution_words)

  remaining_solutions = copy(words)  # List of words that currently fit all known constraints.
  tree = Tree(allowed_guesses, remaining_solutions, nothing, UInt8(0))
  while length(remaining_solutions) > 1
    step = 0
    if length(remaining_solutions) == 3158
      @time while tree.best_choice.best_lower_bound < -3.5526
        add_time(computation_timers.improve, @elapsed improve!(tree, remaining_solutions, allowed_guesses))
        step += 1

        choice = tree.best_choice
        println("Times: ", string(computation_timers))
        println("Step ", step, ": We suggest ", choice)
      end
    else
      @time for _ in 1:1000
        improve!(tree, remaining_solutions, allowed_guesses)
        step += 1

        choice = tree.best_choice
        print(ANSI_RESET_LINE)
        println("Step ", step, ": We suggest ", choice)
      end
    end
    println()

    println("Insert your guess: ")
    guess = Vector{UInt8}(readline(stdin))
    println("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ")
    constraint_template = readline(stdin)
    constraint = parse_constraints(constraint_template)
    # FIXME: build subtree when it does not exist.
    tree = tree.choices[findfirst(c -> c.guess == guess, tree.choices)].constraints[constraint + 1]
    remaining_solutions = filter_solutions_by_constraint(remaining_solutions, guess, constraint)
    println("Remaining words: ", join(map(s -> str_from_word(s), remaining_solutions), ", "))
  end
  println("Solution: ", str_from_word(remaining_solutions[1]), ".")
end

#mutable struct ConvergingMeasurementDifferentials
#  visits::Int
#  max_visits::Int  # Largest number of visits in ConvergingMeasurements.
#  # List of average diffs between each measurement per number of visits.
#  slopes::Vector{Float64}
#  slope_visits::Vector{Int}
#  # Sum of slopes between a given visit count until the latest.
#  biases::Vector{Float64}
#  biases_visits::Vector{Int}
#
#  # Accuracy computation arises from the difference between the start and
#  # converged measurement value.
#  init_slope::Float64  # Average slope of the first half of the data points.
#  # Parameters for an exponential convergence:
#  # estimated_measurement(visits) = asymptote + exp_coeff × exp_base^visits
#  exp_base::Float64
#  exp_coeff::Float64
#  # Precision of the asymptote on a given visits.
#  variance::Vector{Float64}
#  variance_visits::Vector{Int}
#  # y = ax^b, where a = coeff, b = exp, y = variance, x = aggregate visit.
#  variance_coeff::Float64
#  variance_exp::Float64
#end
#
#function ConvergingMeasurementDifferentials()
#  visits = 0
#  max_visits = visits
#  slopes = []
#  slope_visits = []
#  biases = [0]
#  biases_visits = [0]
#  init_slope = 0
#  exp_base = 0
#  exp_coeff = 0
#  variance = [0]
#  variance_visits = [0]
#  variance_coeff = 0
#  variance_exp = 0
#  return ConvergingMeasurementDifferentials(visits, max_visits, slopes, slope_visits, biases, biases_visits, init_slope, exp_base, exp_coeff, variance, variance_visits, variance_coeff, variance_exp)
#end
#
#mutable struct ConvergingMeasurement
#  latest::Float64
#  visits::Int
#  current::Float64  # Smoothed latest value
#  average::Float64
#  # Estimation of where the sequence converges.
#  # Always computed directly from the previous parameters.
#  asymptote::Float64
#  # List of estimated asymptotes for each exploration.
#  asymptotes::Vector{Float64}
#  asymptote_mean::Float64
#  asymptote_variancet::Float64  # Precision of the asymptote, as variance times N
#  current_slope::Float64  # Average slope of the second half.
#  differentials::ConvergingMeasurementDifferentials
#end
#
#function ConvergingMeasurement(differentials::ConvergingMeasurementDifferentials)::ConvergingMeasurement
#  latest = 0
#  visits = 0
#  current = latest
#  average = latest
#  asymptote = latest
#  asymptotes = []
#  asymptote_mean = latest
#  asymptote_variancet = 0
#  current_slope = 0
#  return ConvergingMeasurement(latest, visits, current, average, asymptote, asymptotes, asymptote_mean, asymptote_variancet, current_slope, differentials)
#end
#
#function ConvergingMeasurement()::ConvergingMeasurement
#  return ConvergingMeasurement(ConvergingMeasurementDifferentials())
#end
#
## Visits are 0-indexed.
#function add_measurement!(aggregate::ConvergingMeasurement, new_measurement::Float64)
#  aggregate.visits += 1
#  aggregate.differentials.max_visits = max(aggregate.differentials.max_visits, aggregate.visits)
#  # Update value statistics.
#  old_measurement = aggregate.latest
#  aggregate.latest = new_measurement
#  aggregate.average = (aggregate.average * (aggregate.visits-1) + new_measurement) / aggregate.visits
#  aggregate.current = if aggregate.visits < 2
#    new_measurement
#  else
#    (aggregate.current + new_measurement) / 2
#  end
#
#  update_differentials!(aggregate, new_measurement - old_measurement)
#  #update_exponential_params!(aggregate)
#  add_asymptote!(aggregate)
#end
#
#function update_differentials!(aggregate::ConvergingMeasurement, measured_diff::Float64)
#  if aggregate.visits < 2  # Need 2 points for a diff.
#    return
#  end
#  differentials = aggregate.differentials
#
#  # Update tree-wide slopes.
#  slope_visits = aggregate.visits - 1
#  if slope_visits > length(differentials.slopes)
#    push!(differentials.slopes, 0)
#    push!(differentials.slope_visits, 0)
#    push!(differentials.biases, 0)
#    push!(differentials.biases_visits, 0)
#  end
#  differentials.slope_visits[slope_visits] += 1
#  old_slope = differentials.slopes[slope_visits]
#  new_slope = streamed_mean(old_slope, measured_diff, differentials.slope_visits[slope_visits])
#  differentials.slopes[slope_visits] = new_slope
#
#  # Update biases.
#  nudge = new_slope - old_slope
#  for i = 1:aggregate.visits-1
#    differentials.biases_visits[i] += 1
#    differentials.biases[i] += nudge
#  end
#
#  # Exponential weighing of (½^i)÷2 to smooth the differential.
#  # That way, the last weighs ½; the nearest 5th measurement still weighs >1%.
#  aggregate.current_slope = if aggregate.visits == 2
#    measured_diff
#  else
#    (aggregate.current_slope + measured_diff) / 2
#  end
#
#  # The initial slope is the average of the first diff in differentials.
#  if aggregate.visits == 2
#    differentials.visits += 1
#    differentials.init_slope = (differentials.init_slope * (differentials.visits-1) + measured_diff) / differentials.visits
#  end
#end

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

## Add asymptote estimate for the current number of visits.
## Only trigger this when adding a new measurement,
## and after having incremented aggregate.visits.
#function add_asymptote!(aggregate::ConvergingMeasurement)
#  differentials = aggregate.differentials
#  new_asymptote_value = update_asymptote!(aggregate)
#  push!(aggregate.asymptotes, new_asymptote_value)
#
#  # Update asymptote mean and variance.
#  old_asymptote_mean = aggregate.asymptote_mean
#  old_asymptote_variancet = aggregate.asymptote_variancet
#  new_asymptote_count = aggregate.visits
#  if new_asymptote_count != length(aggregate.asymptotes)
#    println("Invalid new_asymptote_count: ", new_asymptote_count,
#            " while the registered asymptotes are ", aggregate.asymptotes)
#  end
#  new_asymptote_mean = streamed_mean(old_asymptote_mean, new_asymptote_value, new_asymptote_count)
#  aggregate.asymptote_mean = new_asymptote_mean
#  aggregate.asymptote_variancet = streamed_variance_times_count(
#    old_asymptote_variancet, old_asymptote_mean, new_asymptote_mean, new_asymptote_value)
#
#  # If we are the first to register a variance for this visit count, add it.
#  if length(differentials.variance) < aggregate.visits
#    push!(differentials.variance, 0)
#    push!(differentials.variance_visits, 0)
#  end
#  new_variance_value = aggregate.asymptote_variancet / (new_asymptote_count - 1)
#  if !isnan(new_variance_value)
#    new_var_visits = differentials.variance_visits[aggregate.visits] + 1
#    differentials.variance_visits[aggregate.visits] = new_var_visits
#    # We include the updated variance into the tree variance average.
#    old_variance = differentials.variance[aggregate.visits]
#    differentials.variance[aggregate.visits] = streamed_mean(old_variance, new_variance_value, new_var_visits)
#  end
#end
#
#function update_asymptote!(aggregate::ConvergingMeasurement)::Float64
#  aggregate.asymptote = estimate_asymptote(aggregate)
#  return aggregate.asymptote
#end
#
#function estimate_asymptote(aggregate::ConvergingMeasurement)::Float64
#  return estimate_asymptote_from_bias(aggregate)
#end
#
## Estimate the measured asymptote from exponential regression.
#function estimate_asymptote_from_exp_reg(aggregate::ConvergingMeasurement)::Float64
#  differentials = aggregate.differentials
#  # yi = a + bc^i => a = yi - bc^i
#  return aggregate.current - differentials.exp_coeff*differentials.exp_base^aggregate.visits
#end
#
#function estimate_bias(aggregate::ConvergingMeasurement)::Float64
#  differentials = aggregate.differentials
#  bias = 0
#  v = max(aggregate.visits, 1)
#  for i = v:differentials.max_visits-1
#    bias += differentials.slopes[i]
#  end
#  return bias
#end
#
## Compute the bias that measurements incur by summing the averaged diffs between
## each sequential measurement.
#function estimate_asymptote_from_bias(aggregate::ConvergingMeasurement)::Float64
#  return aggregate.latest + aggregate.differentials.biases[aggregate.visits]
#  #differentials = aggregate.differentials
#  #bias = 0
#  #v = max(aggregate.visits, 1)
#  #for i = v:differentials.max_visits-1
#  #  bias += differentials.slopes[i]
#  #end
#  #return aggregate.latest + bias
#end
#
#function estimate_variance(aggregate::ConvergingMeasurement)::Float64
#  if length(aggregate.differentials.variance) < 2
#    return aggregate.asymptote^2
#  elseif aggregate.visits < 2
#    return aggregate.differentials.variance[2]
#  end
#  return aggregate.differentials.variance[aggregate.visits]
#end
#
#function asymptote_variance(aggregate::ConvergingMeasurement)::Float64
#  return estimate_variance(aggregate)
#end
#
#function string(aggregate::ConvergingMeasurement)::String
#  differentials = aggregate.differentials
#  return string("latest=", @sprintf("%.4f", aggregate.latest),
#    " v=", @sprintf("%d", aggregate.visits),
#    " cur=", @sprintf("%.4f", aggregate.current),
#    " avg=", @sprintf("%.4f", aggregate.average),
#    " asym=[", join(map(a -> @sprintf("%.4f", a), aggregate.asymptotes[1:min(4, length(aggregate.asymptotes))]), " "), "]",
#    " asym_mean=", @sprintf("%.4f", aggregate.asymptote_mean),
#    " asym_var=", @sprintf("%.4f", aggregate.asymptote_variancet / aggregate.visits),
#    " cur_slope=", @sprintf("%.4f", aggregate.current_slope),
#    " init_slope=", @sprintf("%.4f", differentials.init_slope),
#    " exp_base=", @sprintf("%.4f", differentials.exp_base),
#    " exp_coeff=", @sprintf("%.4f", differentials.exp_coeff),
#    " maxv=", @sprintf("%d", differentials.max_visits),
#    " dv=", @sprintf("%d", differentials.visits),
#    " dvar=[", join(map(a -> @sprintf("%.4f", a), differentials.variance[1:min(4, length(differentials.variance))]), " "), "]",
#    " dvar_visits=[", join(map(a -> @sprintf("%.4f", a), differentials.variance_visits[1:min(4, length(differentials.variance_visits))]), " "), "]",
#    " dvar_coeff=", @sprintf("%.4f", differentials.variance_coeff),
#    " dvar_exp=", @sprintf("%.4f", differentials.variance_exp))
#end

# Streamed mean and variance of a sequence of values.
# Welford’s online algorithm, as seen in
# Donald Knuth’s Art of Computer Programming, Vol 2, page 232, 3rd edition.

function streamed_mean(old_mean, new_value, new_count)
  return old_mean + (new_value - old_mean) / new_count
end

# The variance is the output divided by the number of samples minus 1,
# in order to avoid numerical errors and biases.
function streamed_variance_times_count(old_variance_t, old_mean, new_mean, new_value)
  return old_variance_t + (new_value-old_mean) * (new_value-new_mean)
end

abstract type AbstractTree end
abstract type AbstractEstimatorStats end
#abstract type AbstractEstimator end
abstract type AbstractActionValue end

mutable struct Choice
  tree::AbstractTree
  guess::Vector{UInt8}

  # Including this guess, how many guesses until the win using the current best
  # policy. It is always a lower bound for the optimal value.
  best_lower_bound::Float64

  value::AbstractActionValue

  # The following probabilities are local estimations;
  # they must be divided by sum_prob_optimal etc. to be coherent.
  prob_optimal::Float64
  #prob_improvement::Float64
  #exploratory_reward::Float64  # Expected improvement in the tree playing reward after exploration.
  # Constraint that Wordle may yield as a response to this guess.
  # We link it lazily to the next choice we make in the tree search.
  visits::Int
  last_visit::Int  # Number of tree visits during last exploration.
  #visits_with_improvement::Int

  #measurement::ConvergingMeasurement
  #reward_estimator::Union{AbstractEstimator, Nothing}

  constraints::Union{Vector{Union{AbstractTree, Nothing}}, Nothing}
end

mutable struct Tree <: AbstractTree
  previous_choice::Union{Choice, Nothing}
  constraint::UInt8
  choices::Vector{Choice}
  choice_from_guess::Dict{Vector{UInt8}, Choice}
  best_choice::Union{Choice, Nothing}
  # Best known choice based on the known lower bound explored.
  best_choice_lower_bound::Union{Choice, Nothing}
  nsolutions::Int
  visits::Int
  #sum_prob_optimal::Float64
  newest_choice::Union{Choice, Nothing}
  #last_non_cache_visit::Int  # Visit count when we last included a guess in the list of choices.
  #differentials::ConvergingMeasurementDifferentials
  estimator_stats::AbstractEstimatorStats
end

function Choice(guess::Vector{UInt8})::Choice
  tree = nothing
  best_lower_bound = -1
  value = ActionValue(-1, -1, -1)
  prob_optimal = 1
  #prob_improvement = 0
  #exploratory_reward = 0
  visits = 0
  last_visit = -1
  #visits_with_improvement = 0
  #reward_estimator = nothing
  constraints = nothing
  Choice(tree, guess, best_lower_bound, value, prob_optimal, visits, last_visit, constraints)
end

function Choice(tree::Tree, guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}}, value_estimate::Float64)::Choice
  nsols = Float64(length(solutions))
  best_lower_bound = -nsols
  value = ActionValue(value_estimate, tree)
  prob_optimal = 1
  #prob_improvement = 1
  #exploratory_reward = 1
  visits = 0
  last_visit = -1
  #visits_with_improvement = 0
  #reward_estimator = Estimator(tree.estimator_stats)
  #add_estimate!(reward_estimator, value_estimate)
  constraints = nothing
  return Choice(tree, guess, best_lower_bound, value, prob_optimal, visits, last_visit, constraints)
end

function Base.show(io::IO, choice::Choice)
  print(io, str_from_word(choice.guess), " ",
    @sprintf("%.4f", -choice.best_lower_bound), ">",
    @sprintf("%.4f", -choice.value.estimate), "→",
    @sprintf("%.4f", -choice.value.tree_estimate), "→",
    @sprintf("%.4f", -choice.value.debiased), "±",
    @sprintf("%.4f", sqrt(debiased_variance(choice))), "; ",
    @sprintf("o=%d%%", round(choice.prob_optimal * 100)), "; ",
    "lv=", choice.last_visit, "; ",
    "v=", choice.visits)
end

function Tree(guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}}, previous_choice::Union{Choice, Nothing}, constraint::UInt8)::Tree
  nsols = length(solutions)
  if nsols == 1
    choice = Choice(solutions[1])
    best_choice = choice
    best_choice_lower_bound = choice
    visits = 0
    sum_prob_optimal = choice.prob_optimal
    newest_choice = nothing
    last_non_cache_visit = -1
    estimator_stats = EstimatorStats()
    tree = Tree(previous_choice, constraint, [choice], Dict([(choice.guess, choice)]), best_choice, best_choice_lower_bound, nsols, visits, newest_choice, estimator_stats)
    choice.tree = tree
    estimator_stats.tree = tree
    return tree
  end

  visits = 0
  sum_prob_optimal = 0
  newest_choice = nothing
  last_non_cache_visit = -1
  estimator_stats = EstimatorStats()
  tree = Tree(previous_choice, constraint, [], Dict{Vector{UInt8}, Choice}(), nothing, nothing, nsols, visits, newest_choice, estimator_stats)
  estimator_stats.tree = tree
  add_choice_from_best_uncached_action!(tree, guesses, solutions)
  if isnothing(tree.best_choice)
    error(string("Tree: error: no best choice found, in ", choice_breadcrumb(tree.previous_choice)))
  end
  #update_prob_explore!(tree)
  return tree
end


# We assume that measurements follow a common statistical distribution for the
# same amount of exploration, with respect to how off they are from the
# asymptotic measurement. We estimate the distribution based on all the measured
# deltas (for a given exploration count) between the initial estimate measured,
# and the expected value of the asymptotic measurement after an infinite number
# of explorations.

mutable struct EstimatorStats <: AbstractEstimatorStats
  # Number of actions that have done at least N-1 visits.
  actions_with_visits::Vector{Int}
  # For each estimate count, we keep track of the mean bias of the estimator.
  # In other words, the average difference between an estimation with N-1
  # explorations and one with N, the latter being assumed as more precise.
  visit_bias::Vector{Float64}
  # The overall bias from one estimate count, to the latest estimate.
  bias::Vector{Float64}
  # Variance of the difference between the debiased estimate after the Nth visit
  # and the N-1 visit, times the number of samples. To allow streaming computation.
  debiased_delta_variance_t::Vector{Float64}
  # The variance of the debiased estimator after N-1 visits.
  # It represents the uncertainty of its value,
  # thus works like a mean squared error compared to the true action value.
  debiased_variance::Vector{Float64}
  #init_variance::Float64
  tree::Union{Tree, Nothing}
end

EstimatorStats() = EstimatorStats([], [], [], [], [], nothing)

function Base.show(io::IO, stats::EstimatorStats)
  println(io, "visits\tactions_with_visits\tvisit_bias\tbias\tdebiased_delta_variance_t\tdebiased_variance")
  for i = 1:length(stats.bias)
    println(io, @sprintf("%d\t%d\t%.4f\t%.4f\t%.15f\t%.15f",
      i-1,
      stats.actions_with_visits[i],
      stats.visit_bias[i],
      stats.bias[i],
      stats.debiased_delta_variance_t[i],
      stats.debiased_variance[i]))
  end
end

#function compute_mean_squared_error!(stats::EstimatorStats)
#  for exploration_count in 1:length(stats.mean_squared_error)
#    sum = 0
#    count = 0
#    for choice in stats.tree.choices
#      if exploration_count > length(choice.reward_estimator.values)
#        continue
#      end
#      estimate = choice.reward_estimator.values[exploration_count]
#      real = choice.reward_estimator.debiased
#      sum += (real - estimate)^2
#      count += 1
#    end
#    stats.mean_squared_error[exploration_count] = sum / count
#  end
#end
#
#function compute_variance!(stats::EstimatorStats)
#  for exploration_count in 1:length(stats.variance)
#    stats.variance[exploration_count] = stats.mean_squared_error[exploration_count] #- stats.bias[exploration_count]^2
#  end
#  for choice in stats.tree.choices
#    choice.reward_estimator.variance = variance(choice.reward_estimator)
#  end
#end

#function compute_init_variance!(stats::EstimatorStats)
#  sum = 0
#  count = 0
#  for choice in stats.tree.choices
#    if length(choice.reward_estimator.values) >= 2
#      sum += choice.reward_estimator.variance
#      count += 1
#    end
#  end
#  stats.init_variance = sum / count
#  for choice in stats.tree.choices
#    if length(choice.reward_estimator.values) < 2
#      choice.reward_estimator.variance = stats.init_variance #variance(choice.reward_estimator)
#    end
#  end
#end
#
#function append_exploration!(stats::EstimatorStats)
#  push!(stats.visits, 0)
#  push!(stats.visit_bias, 0)
#  push!(stats.bias, 0)
#end
#
#mutable struct Estimator <: AbstractEstimator
#  # Each recorded value is a play reward for a different exploration.
#  values::Vector{Float64}
#  mean::Float64
#  # Estimates variance. Beware: it is not the estimator variance.
#  variance_t::Float64
#  variance::Float64
#  # Debiased mean.
#  debiased::Float64
#  stats::EstimatorStats
#end
#
#Estimator(stats::EstimatorStats) = Estimator([], 0, 0, 0, 0, stats)
#
#function variance(estimator::Estimator)::Float64
#  if length(estimator.values) == 0
#    return estimator.stats.init_variance
#  else
#    return estimator.variance_t / length(estimator.values)
#  end
#end
#
#function debiased(estimator::Estimator)::Float64
#  return estimator.mean + estimator.stats.bias[length(estimator.values)]
#end
#
#function add_estimate!(estimator::Estimator, value::Float64)
#  push!(estimator.values, value)
#  estimate_count = length(estimator.values)
#
#  while length(estimator.stats.visit_bias) < estimate_count
#    append_exploration!(estimator.stats)
#  end
#
#  old_mean = estimator.mean
#  old_variance_t = estimator.variance_t
#  estimator.mean = streamed_mean(old_mean, value, estimate_count)
#  estimator.variance_t = streamed_variance_times_count(old_variance_t, old_mean, estimator.mean, value)
#  estimator.variance = variance(estimator)
#  estimator.debiased = debiased(estimator)
#
#  if estimate_count > 1
#    # Update statistical bias between the two exploration counts.
#    visit_bias_sample = value - estimator.values[estimate_count-1]
#    estimator.stats.visits[estimate_count-1] += 1
#
#    old_visit_bias = estimator.stats.visit_bias[estimate_count-1]
#    new_visit_bias = streamed_mean(old_visit_bias, visit_bias_sample, estimator.stats.visits[estimate_count-1])
#    estimator.stats.visit_bias[estimate_count-1] = new_visit_bias
#  end
#
#  # Update bias from exploration counts before this one and the latest one.
#  if estimate_count == length(estimator.stats.bias)
#    estimator.stats.bias[estimate_count] = estimator.stats.visit_bias[estimate_count]
#  end
#  for i in estimate_count-1:-1:1
#    estimator.stats.bias[i] = estimator.stats.bias[i+1] + estimator.stats.visit_bias[i]
#  end
#
#  compute_init_variance!(estimator.stats)
#end
#
#function string(estimator::Estimator)
#  visit_bias = estimator.stats.visit_bias
#  bias = estimator.stats.bias
#  return string(
#    @sprintf(" mea=%.4f", estimator.mean),
#    @sprintf(" deb=%.4f", estimator.debiased),
#    @sprintf(" dev=%.4f", sqrt(estimator.variance)),
#    " val=", join(map(v -> @sprintf("%.4f", v), estimator.values[1:min(10, length(estimator.values))]), ","), "…",
#    " vbi=", join(map(v -> @sprintf("%.4f", v), visit_bias[1:min(10, length(visit_bias))]), ","), "…",
#    " bia=", join(map(v -> @sprintf("%.4f", v), bias[1:min(10, length(bias))]), ","), "…",
#    " ini=", @sprintf("%.4f", sqrt(estimator.stats.init_variance)),
#  )
#end

# Estimated cumulative reward when choosing this action with an optimal policy.
mutable struct ActionValue  <: AbstractActionValue
  estimate::Float64  # Baseline estimator (biased, unless in endgame).
  tree_estimate::Float64  # Estimator using children action values (biased).
  debiased::Float64  # Debiased tree estimator.
end

function ActionValue(estimate::Float64, tree::Tree)::ActionValue
  return ActionValue(estimate, estimate, debiased(estimate, 0, tree))
end

function update_action_value!(choice::Choice, tree_estimate::Float64)
  choice.visits += 1
  choice.last_visit = choice.tree.visits
  choice.tree.visits += 1
  update_tree_stats!(choice, tree_estimate)
  choice.value.tree_estimate = tree_estimate
  choice.value.debiased = debiased(choice)
end

function debiased(choice::Choice)::Float64
  return debiased(choice.value.tree_estimate, choice.visits, choice.tree)
end

function debiased(value_estimate::Float64, visits::Int, tree::Tree)::Float64
  return value_estimate + bias_from_visit_to_end(tree, visits)
end

function bias_from_visit_to_end(tree::Tree, visits::Int)::Float64
  # If there are not enough entries in the bias
  # (because the current visit count is the max one,
  # so there is no delta to the next visit estimate),
  # we consider there is no bias.
  if visits+1 > length(tree.estimator_stats.bias)
    return 0
  end
  return tree.estimator_stats.bias[visits+1]
end

function bias_variance_from_visit_to_end(tree::Tree, visits::Int)::Float64
  # If there are not enough entries in the bias
  # (because the current visit count is the max one,
  # so there is no delta to the next visit estimate),
  # we consider the latest bias.
  last_idx = lastindex(tree.estimator_stats.debiased_variance)
  if last_idx == 0
    return tree.best_choice.value.debiased^2
  end
  return tree.estimator_stats.debiased_variance[min(visits+1, last_idx)]
end

function debiased_variance(choice::Choice)::Float64
  return bias_variance_from_visit_to_end(choice.tree, choice.visits)
end

# Given a new action value estimate,
# we update the tree-level statistics associated with them.
function update_tree_stats!(choice::Choice, new_tree_estimate::Float64)
  old_tree_estimate = choice.value.tree_estimate
  tree = choice.tree
  resize_tree_stats!(tree, choice.visits)
  tree.estimator_stats.actions_with_visits[choice.visits+1] += 1
  update_tree_visit_bias!(choice, old_tree_estimate, new_tree_estimate)
  update_tree_bias!(tree)

  old_debiased_estimate = choice.value.debiased
  new_debiased_estimate = debiased(new_tree_estimate, choice.visits, choice.tree)
  update_tree_bias_variance!(choice, old_debiased_estimate, new_debiased_estimate)
end

# Ensure that the stats vectors have enough slots to avoid out-of-bound errors.
function resize_tree_stats!(tree::Tree, visits::Int)
  stats = tree.estimator_stats
  actions_with_visits = stats.actions_with_visits
  visit_bias = stats.visit_bias
  bias = stats.bias
  debiased_delta_variance_t = stats.debiased_delta_variance_t
  debiased_variance = stats.debiased_variance
  while length(actions_with_visits) < visits+1
    push!(actions_with_visits, 0)
  end
  while length(visit_bias) < visits
    push!(visit_bias, 0)
  end
  while length(bias) < visits
    push!(bias, 0)
  end
  while length(debiased_delta_variance_t) < visits
    push!(debiased_delta_variance_t, 0)
  end
  while length(debiased_variance) < visits
    push!(debiased_variance, 0)
  end
end

function update_tree_visit_bias!(choice::Choice, old_action_value::Float64, action_value::Float64)
  visit_bias = choice.tree.estimator_stats.visit_bias
  v = choice.visits
  action_count = choice.tree.estimator_stats.actions_with_visits[v]
  visit_bias[v] = streamed_mean(visit_bias[v], action_value - old_action_value, action_count)
end

function update_tree_bias!(tree::Tree)
  # Now, we use those consecutive biases to built biases to the end.
  visit_bias = tree.estimator_stats.visit_bias
  bias = tree.estimator_stats.bias
  # We need to reset the bias.
  fill!(bias, 0)
  # The loop will not set the last value, so we must do so explicitly.
  bias[lastindex(bias)] = visit_bias[lastindex(bias)]
  # The following loop uses a recurrence but works from last to first.
  for visits = lastindex(bias)-2:-1:0
    bias[visits+1] = visit_bias[visits+1] + bias[visits+2]
  end

  # We have the biases, and can now recompute the choices’ debiased value.
  for choice in tree.choices
    choice.value.debiased = debiased(choice)
  end
end

function update_tree_bias_variance!(choice::Choice, old_debiased_estimate::Float64, new_debiased_estimate::Float64)
  stats = choice.tree.estimator_stats
  debiased_delta_variance_t = stats.debiased_delta_variance_t
  debiased_variance = stats.debiased_variance
  v = choice.visits
  action_count = choice.tree.estimator_stats.actions_with_visits[v]

  # The estimator for the variance of the debiased action value estimate
  # after n visits, across actions A is:
  # Σa∈A (debiased(a, n) - action_value(a))² ÷ (a-1)
  # = Σa (action_value(a) - Σ[k=n→∞] Δdebiased(a) - action_value(a))² ÷ (a-1)
  #   where Δdebiased(a) = debiased(a, n+1) - debiased(a, n)
  #   because debiased(a, n) converges to action_value(a) as n → ∞
  #   thus debiased(a, n) + Σ[k=n→∞] (debiased(a, k+1) - debiased(a, k)) = action_value(a)
  # = var(ΣΔdebiased)  since 𝔼Δdebiased = 0 as it is debiased.
  # = Σvar(Δdebiased)  since they are mostly uncorrelated.
  delta = new_debiased_estimate - old_debiased_estimate
  debiased_delta_variance_t[v] += delta^2

  # Reset the variance to match the delta variance.
  for v = 1:length(debiased_variance)
    action_count = choice.tree.estimator_stats.actions_with_visits[v]
    # Unbiased estimator for the variance, using the Bessel correction.
    # Since the count is action_count+1,
    # removing 1 for Bessel correction yields action_count:
    debiased_variance[v] = if action_count > 1
      debiased_delta_variance_t[v] / (action_count - 1)
    else  # To avoid a division by zero, we don’t use Bessel correction.
      debiased_delta_variance_t[v]
    end
  end

  # The real debiased variance sums the delta variances.
  for v = length(debiased_variance)-1:-1:1
    debiased_variance[v] += debiased_variance[v+1]
  end
end

#function update_biases!(tree::Tree)
#  # We first need to recompute the visit bias (between consecutive visits).
#  visit_bias = tree.estimator_stats.visit_bias
#  for visits = 0:length(visit_bias)-1
#    sum = 0.0
#    count = 0
#    for choice in tree.choices
#      estimates = choice.value.estimates
#      if length(estimates) >= visits+2
#        sum += estimates[visits+2] - estimates[visits+1]
#        count += 1
#      end
#    end
#    visit_bias[visits+1] = sum / count
#  end
#
#  # Now, we use those consecutive biases to built biases to the end.
#  visit_bias = tree.estimator_stats.visit_bias
#  bias = tree.estimator_stats.bias
#  # We need to reset the bias.
#  fill!(bias, 0)
#  # The loop will not set the last value, so we must do so explicitly.
#  bias[lastindex(bias)] = visit_bias[lastindex(bias)]
#  # The following loop uses a recurrence but works ontil till the last.
#  for visits = lastindex(bias)-2:-1:0
#    bias[visits+1] = visit_bias[visits+1] + bias[visits+2]
#  end
#
#  # We have the biases, and can now recompute the choices’ debiased value.
#  for choice in tree.choices
#    choice.value.debiased = debiased(choice)
#  end
#
#  # Now that we can compare biased estimates to their debiased counterpart,
#  # let’s use that to compute the variance of the bias.
#  debiased_variance = tree.estimator_stats.debiased_variance
#  for visits = 0:length(debiased_variance)-1
#    sum = 0.0
#    count = 0
#    for choice in tree.choices
#      estimates = choice.value.estimates
#      if length(estimates) >= visits+1
#        sum += (estimates[visits+1] - bias[visits+1] - choice.value.debiased)^2
#        count += 1
#      end
#    end
#    debiased_variance[visits+1] = sum / count
#  end
#end


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
    choice, choice_idx = best_exploratory_choice!(tree, solutions, guesses)
  end)
  #init_exploratory_reward = choice.exploratory_reward
  #init_estimate_latest = choice.reward_estimator.mean
  init_best_lower_bound = choice.best_lower_bound
  #if nsolutions == 2315
  #  println("Before exploration: ", string(choice.reward_estimator))
  #end
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
        subtree = Tree(guesses, remaining_solutions, choice, constraint)
        choice.constraints[constraint + 1] = subtree
      end)
    else
      improve!(subtree, remaining_solutions, guesses)
    end

    ## Full-depth:
    #if isnothing(subtree)        # Initialize the next move.
    #  add_time(computation_timers.new_tree, @elapsed begin
    #    subtree = Tree(guesses, remaining_solutions, choice, constraint)
    #    choice.constraints[constraint + 1] = subtree
    #  end)
    #end
    #improve!(subtree, remaining_solutions, guesses)

    # For each solution, we made one guess, on top of the guesses left to end the game.
    best_guesses_to_win += (subtree.best_choice.best_lower_bound - 1) * nrsols
    # FIXME: we should not use the most optimistic estimate,
    # but the expected estimate,
    # by weighing each expected play reward by the probability that it is optimal.
    new_tree_optimal_estimate += (subtree.best_choice.value.debiased - 1) * nrsols
    #new_tree_optimal_estimate += (subtree.best_choice.reward_estimator.mean - 1) * nrsols
  end

  # Update information about the current best policy.
  new_guesses_remaining = best_guesses_to_win / nsolutions
  new_tree_optimal_estimate /= nsolutions
  add_time(computation_timers.add_measurement, @elapsed begin
    update_action_value!(choice, new_tree_optimal_estimate)
  end)
  #add_time(computation_timers.update_prob_explore, @elapsed begin
  #  update_prob_explore!(tree)
  #end)
  update_best_choices!(choice, choice_idx, new_guesses_remaining)
  if nsolutions == 2315
  #if !isnothing(findfirst(s -> str_from_word(s) == "vowel", solutions))
    #println("First choice optimal prob: ", tree.choices[1].prob_optimal)
    println("Explored ", choice_breadcrumb(choice), ": ", nsolutions, " sols (",
            @sprintf("%.4f", -init_best_lower_bound), "→",
            @sprintf("%.4f", -choice.best_lower_bound), ") ",
            choice)
            #@sprintf("%.4f", -init_estimate_latest), "→",
            #@sprintf("%.4f", -choice.value.debiased), "±",
            #@sprintf("%.4f", sqrt(debiased_variance(choice))), "; ",
            #@sprintf("%.4f", -choice.reward_estimator.mean), "∞→",
            #@sprintf("%.4f", -choice.reward_estimator.debiased), "±",
            #@sprintf("%.4f", sqrt(choice.reward_estimator.variance)), ";",
            #@sprintf("e=%d", init_exploratory_reward), "→",
            #@sprintf("%d", choice.exploratory_reward), ";",
            #@sprintf("o=%d%%", round(choice.prob_optimal * 100)), "; ",
            #@sprintf("i=%d%%", round(choice.prob_improvement * 100)), ";",
            #"v=", choice.visits, ")")
    #println("After exploration: ", string(choice.reward_estimator))
    println()
  end
end

# Pick the choice that is most valuable to explore.
function best_exploratory_choice!(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
  return choice_from_thompson_sampling!(tree, solutions, guesses)
  # Ablation study: using exploratory reward yields too much sensitivity to
  # optimal choices incorrectly assessed as unimprovable.
  #return choice_with_max_expected_exploratory_reward(tree, solutions, guesses)
end

# Select a choice in proprotion to the probability that it is optimal.
function choice_from_thompson_sampling!(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
  update_prob_optimal!(tree)
  return probabilist_thompson_sample(tree, solutions, guesses)
end

function update_prob_optimal!(tree::Tree)
  # We perform a set of Monte Carlo simulations
  # to statistically estimate how often a choice comes on top.
  opt_count = zeros(Int, length(tree.choices))
  samples = 1024
  for _ = 1:samples
    best_idx = 0
    best_action_value = -Inf
    for (i, choice) in enumerate(tree.choices)
      action_value = sample_action_value(choice)
      if action_value > best_action_value
        best_action_value = action_value
        best_idx = i
      end
    end
    opt_count[best_idx] += 1
  end

  # The optimal probability is the ratio that this choice was optimal.
  for (i, choice) in enumerate(tree.choices)
    choice.prob_optimal = opt_count[i] / samples
  end
end

# Randomly pick an action value following a Gumbel distribution
# using the debiased tree estimator as mode and the bias variance.
function sample_action_value(choice::Choice)::Float64
  mode = choice.value.debiased
  variance = bias_variance_from_visit_to_end(choice.tree, choice.visits)
  if variance == 0
    #println("Warning: sample_action_value: zero variance for choice ", choice)
    return mode
  end
  # Ablation study: a Gaussian increases the optimal prob of the top choice
  # yet it ensures the next top choices have more visits as well
  # (despite them having a lower optimal proability).
  # The Gumbel suggests the best choice as optimal after 80 steps
  # instead of 96 for the Normal. However:
  # ┌─────────────┬───────────────────────┐
  # │ Lower bound │  Number of steps to   │
  # │             │ find that lower bound │
  # │             ├──────────┬────────────┤
  # │             │ Gaussian │   Gumbel   │
  # ├─────────────┼──────────┼────────────┤
  # │      3.5535 │      752 │       2864 │
  # │      3.5532 │     1744 │       5040 │
  # │      3.5529 │     8336 │        OOM │
  # └─────────────┴──────────┴────────────┘
  return rand(Normal(mode, sqrt(variance)))
  #scale = sqrt(variance * (6/pi^2))
  #return rand(Gumbel(mode, scale))
end

# Pick choices so that the frequency that they are picked at, matches their
# probability of being the optimal action.
function probabilist_thompson_sample(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
  # The set of probabilities is a pie chart,
  # and on this pie wheel, we randomly run a pointer.
  rand_pointer = rand()
  cum_prob = 0.0
  for (choice_idx, choice) in enumerate(tree.choices)
    cum_prob += choice.prob_optimal
    if rand_pointer <= cum_prob
      if isnothing(tree.previous_choice)
        println("Selected ", choice, " (rand_pointer=", rand_pointer, ")")
        println(tree)
      end
      # If we pick the newest choice, we uncache a choice.
      if choice == tree.newest_choice
        add_choice_from_best_uncached_action!(tree, guesses, solutions)
      end
      return choice, choice_idx
    end
  end
  # If a choice has not been selected yet, we pick the newest one.
  choice_idx = length(tree.choices)
  choice = tree.choices[choice_idx]
  # If we pick the newest choice, we uncache a choice.
  if choice == tree.newest_choice
    add_choice_from_best_uncached_action!(tree, guesses, solutions)
  end
  return choice, choice_idx
end

# Pick choices so that the frequency that they are picked at, matches their
# probability of being the optimal action.
#function frequentist_thompson_sample(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
#  for (i, choice) in enumerate(tree.choices)
#    if choice.prob_optimal * (tree.visits - choice.last_visit) >= 1
#      if isnothing(tree.previous_choice)
#        println("Selected ", choice)
#        println(tree)
#      end
#      # If we pick the newest choice, we uncache a choice.
#      if choice == tree.newest_choice
#        add_choice_from_best_uncached_action!(tree, guesses, solutions)
#      end
#      return choice, i
#    end
#  end
#  # If a choice has not been selected yet, we pick the newest one.
#  idx = length(tree.choices)
#  choice = tree.choices[idx]
#  # If we pick the newest choice, we uncache a choice.
#  if choice == tree.newest_choice
#    add_choice_from_best_uncached_action!(tree, guesses, solutions)
#  end
#  return choice, idx
#end

# Pick the choice based on fair total expanded work:
# a choice that has a 40% chance of being optimal
# should be explored until 40% of all historical explorations is theirs.
# We assume that the sum of prob_optimal across choices is 1.
#function fair_thompson_sample(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
#  for (i, choice) in enumerate(tree.choices)
#    if choice.visits < tree.visits * choice.prob_optimal
#      if isnothing(tree.previous_choice)
#        println("Selected ", choice)
#        println(tree)
#      end
#      # If we pick the newest choice, we uncache a choice.
#      if choice == tree.newest_choice
#        add_choice_from_best_uncached_action!(tree, guesses, solutions)
#      end
#      return choice, i
#    end
#  end
#  # If a choice has not been selected yet, we pick the newest one.
#  idx = length(tree.choices)
#  choice = tree.choices[idx]
#  # If we pick the newest choice, we uncache a choice.
#  if choice == tree.newest_choice
#    add_choice_from_best_uncached_action!(tree, guesses, solutions)
#  end
#  return choice, idx
#end


# Return the choice (and its index in the cache) that will produce the highest
# expected exploratory reward if explored.
#function choice_with_max_expected_exploratory_reward(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
#  if isnothing(tree.previous_choice)
#    println("Selection of choice to explore.")
#    for choice in tree.choices
#      println("- Action ", str_from_word(choice.guess), ": exploratory_reward=", choice.exploratory_reward)
#    end
#  end
#
#  max_exploratory_reward = -Inf
#  argmax_choice = nothing
#  argmax_idx = 1
#  for (i, choice) in enumerate(tree.choices)
#    if choice.exploratory_reward >= max_exploratory_reward
#      max_exploratory_reward = choice.exploratory_reward
#      argmax_choice = choice
#      argmax_idx = i
#    end
#  end
#  # If we pick the newest choice, we uncache a choice.
#  if argmax_choice == tree.newest_choice || isnothing(argmax_choice)
#    add_choice_from_best_uncached_action!(tree, guesses, solutions)
#  end
#  return argmax_choice, argmax_idx
#end

# The exploratory reward is the difference between the current best play reward
# (using the expected play reward of the choice that has the highest), and what
# the best play reward is expected to be after exploring this action further.
#
# We use the expected value for the probabiity distribution of the exploration
# reward. We build the latter from the probabiity distribution of play reward
# for each choice.
#
# - For the best choice, it starts at a negative value: the difference between
#   the mode of the best choice, to the mode of the second best, for play
#   rewards below the second best. It grows linearly up to zero for the current
#   mode of the exploration reward, and continues to grow linearly in the
#   positives.
# - For the others, it is zero for all cases where the play reward remains below
#   the mode of the best, and grows linearly above.
#         ↑ Prob.
#         │⠒⠄
#  ⠸    ⠴⠋│ ⠈⠙
#  ⠸   ⠴⠁ │   ⠉⠑⠒⠢⠤⠄
#  ⠸ ⠠⠎   │        ⠈⠑⠒⠢⠤⠄
#  ⠸⠊⠁    │             ⠈⠑⠒⠢⠤⠤⠤⠤⠤⠤⠤⠤⠤
# ────────┼──────────────────────────────────→  Expl. reward
#function exploratory_reward(choice::Choice)::Float64
#  samples = 64
#  sum = 0
#  for _ in 1:samples
#    sum += sample_exploratory_reward(choice::Choice)
#  end
#  return sum / samples
#end
#
#function sample_exploratory_reward(choice::Choice)::Float64
#  mean = choice.reward_estimator.debiased
#  variance = choice.reward_estimator.variance
#  if variance == 0
#    return 0  # We cannot improve on what is determined.
#  end
#  scale = sqrt(variance * (6/pi^2))
#  mode = mean - scale * MathConstants.eulergamma
#
#  new_play_reward = rand(Gumbel(mode, scale))
#  current_best_play_reward = choice.tree.best_choice.reward_estimator.debiased
#  new_best_play_reward = if choice == choice.tree.best_choice
#    max(new_play_reward, choice.tree.second_best_choice.reward_estimator.debiased)
#  else
#    max(new_play_reward, current_best_play_reward)
#  end
#  improvement = new_best_play_reward - current_best_play_reward
#  return improvement
#end

#function choice_that_beats_best(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
#  # What is the cached choice whose frequency elects it first?
#  cached_choice, cached_idx, cached_next_visit = best_cached_exploratory_choice(tree)
#  if isnothing(cached_choice)
#    selection = best_non_cached_exploratory_choice(tree, guesses, solutions)
#    if isnothing(selection)
#      error(string("best_exploratory_choice: error: no cached nor uncached choices, in ", choice_breadcrumb(tree.previous_choice)))
#    end
#    if isnothing(tree.previous_choice)
#      println("\nChose because no cached choice was available")
#    end
#    return selection
#  end
#  # What about the non-cached choices?
#  uncached_exploration_prob = tree.prob_uncached_beat_best / tree.sum_prob_beat_best
#  non_cached_visit_freq = 1 / uncached_exploration_prob
#  non_cached_next_visit = tree.last_non_cache_visit + non_cached_visit_freq
#  if cached_next_visit <= non_cached_next_visit
#    if isnothing(tree.previous_choice)
#      println("\nChose based on cached next visit: ", @sprintf("%.2f", cached_next_visit))
#    end
#    return cached_choice, cached_idx
#  end
#  selection = best_non_cached_exploratory_choice(tree, guesses, solutions)
#  if isnothing(selection)
#    if isnothing(tree.previous_choice)
#      println("\nChose based on cached next visit as there are non uncached choices: ", @sprintf("%.2f", cached_next_visit))
#    end
#    return cached_choice, cached_idx
#  end
#  if isnothing(tree.previous_choice)
#    println("\nChose based on non-cached next visit: ", @sprintf("%.2f", non_cached_next_visit))
#  end
#  return selection
#end
#
## Returns the choice, its index in the cache, and the next tree visit at which
## it needed to be explored.
#function best_cached_exploratory_choice(tree::Tree)::Tuple{Union{Choice, Nothing}, Int, Float64}
#  min_next_visit = Inf
#  min_choice = nothing
#  min_idx = 0
#  for (i, choice) in enumerate(tree.choices)
#    # The frequency of visits should match the probability of exploration.
#    next_visit = if choice.visits <= 0
#      0.0
#    else
#      prob_visit = choice.prob_beat_best / tree.sum_prob_beat_best
#      visit_freq = 1 / prob_visit
#      choice.last_visit + visit_freq
#    end
#    if next_visit < min_next_visit
#      min_next_visit = next_visit
#      min_choice = choice
#      min_idx = i
#    end
#  end
#  return min_choice, min_idx, min_next_visit
#end
#
#function best_non_cached_exploratory_choice(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Union{Tuple{Choice, Int}, Nothing}
#  # Find the remaining guess with the best estimate.
#  choice = add_choice_from_best_uncached_action!(tree, guesses, solutions)
#  if isnothing(choice)
#    return nothing
#  end
#  tree.last_non_cache_visit = tree.visits
#  return choice, length(tree.choices)
#end
#
## Pick the choice based on fair total expanded work: a choice that should be
## explored at 40% should be explored until 40% of all historical explorations
## is theirs.
#function fair_exploratory_choice(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
#  for (i, choice) in enumerate(tree.choices)
#    prob_visit = choice.prob_beat_best / tree.sum_prob_beat_best
#    if choice.visits < tree.visits * prob_visit
#      return choice, i
#    end
#  end
#  selection = best_non_cached_exploratory_choice(tree, guesses, solutions)
#  if isnothing(selection)
#    tree.choices[1], 1
#  end
#  return selection
#end

#function add_measurement!(choice::Choice, new_measurement::Float64)
#  old_reward = choice.reward_estimator.debiased
#  add_estimate!(choice.reward_estimator, new_measurement)
#  if choice.reward_estimator.debiased > old_reward
#    choice.visits_with_improvement += 1
#  end
#
#  #old_asymptote = choice.measurement.asymptote
#  #add_measurement!(choice.measurement, new_measurement)
#  #if choice.measurement.asymptote > old_asymptote
#  #  choice.visits_with_improvement += 1
#  #end
#
#  # Recursive constant:
#  # tree.best_choice.best_lower_bound matches all recursive best choices.
#  # We verify it through assertion testing.
#  #if nsolutions == 2315
#  #  total = total_guesses_for_all_sols(tree, guesses, solutions)
#  #  if abs(tree.best_choice.best_lower_bound - total / nsolutions) > 0.01
#  #    println("Total guesses: ", total, "; tree best choice guesses: ", tree.best_choice.best_lower_bound)
#  #  end
#  #end
#
#  # For exploration, we rely on a combined metric that estimates the lower bound
#  # of the number of guesses left before winning, based on our uncertainty.
#  choice.visits += 1
#  choice.tree.visits += 1
#end

function update_best_choices!(choice::Choice, choice_idx::Int, new_lower_bound::Float64)
  tree = choice.tree
  tree.best_choice = tree.choices[argmax(choice.value.debiased for choice in tree.choices)]

  old_tree_best_lower_bound = tree.best_choice_lower_bound.best_lower_bound
  if new_lower_bound > choice.best_lower_bound
    choice.best_lower_bound = new_lower_bound
  end
  if choice.best_lower_bound > old_tree_best_lower_bound
    println("Improvement found: ", choice_breadcrumb(choice), " ",
            @sprintf("%.4f", old_tree_best_lower_bound), "→",
            @sprintf("%.4f", choice.best_lower_bound),
            " (rank ", choice_idx, "; ", tree.nsolutions, " sols)")
    tree.best_choice_lower_bound = choice
  end
end

## The likelihood that we pick a choice is its worthiness:
## the odds that it is optimal and that its exploration improves its score.
#function update_prob_explore!(tree::Tree)
#  for c in tree.choices
#    c.exploratory_reward = exploratory_reward(c)
#  end
#end

## Probability that this choice is optimal under perfect play.
#function prob_optimal_choice(optimal_estimate::Float64, optimal_estimate_variance::Float64, tree::Tree)::Float64
#  if isinf(optimal_estimate)
#    return 0
#  end
#  prob = 1
#  for other in tree.choices
#    prob *= prob_superior_choice(optimal_estimate, optimal_estimate_variance, other)
#  end
#  return prob
#end
#
#function prob_superior_choice(optimal_estimate::Float64, optimal_estimate_variance::Float64, other::Choice)::Float64
#  diff_variance = optimal_estimate_variance + asymptote_variance(other.measurement)
#  if diff_variance == 0
#    if optimal_estimate >= other.measurement.asymptote
#      return 1
#    else
#      return 0
#    end
#  end
#  # We now pretend the difference between this choice’s distribution
#  # and the best choice’s is logistic.
#  choice_mu = optimal_estimate  # We consider it to be the mode of its Gumbel.
#  other_mu = other.measurement.asymptote
#  return 1 - 1 / (1 + exp(-(0-(choice_mu-other_mu))/(sqrt(3*diff_variance)/pi)))
#end
#
## Alternative estimate of the probability of a choice being optimal.
#function prob_choice_reaching_optimal(optimal_estimate::Float64, optimal_estimate_variance::Float64, tree::Tree)::Float64
#  # We estimate it by comparing the probability that a choice reaches the
#  # current overall optimal estimate, under the categorical distribution.
#  optimum = if isnothing(tree.best_choice)
#    optimal_estimate
#  else
#    tree.best_choice.measurement.asymptote
#  end
#  # We assume that optimal estimates follow a Gumbel distribution,
#  # since it is the maximum of a set of current-best-measurements
#  # which follow an exponential distribution.
#  mode = optimal_estimate
#  # FIXME: we are comparing optimality probabilities between choices which had a
#  # large number of guesses, improving their variance, and choices that did not.
#  # Ideally, they should be compared with a variance that they would have after
#  # the same number of visits.
#  beta = sqrt(optimal_estimate_variance * 6 / pi^2)
#  z = (optimum - mode) / beta
#  return exp(-(z + exp(-z))) / beta
#end
#
## Probability that the given choice will surpass the best asymptote.
#function prob_beat_best(choice::Choice)::Float64
#  return prob_beat_best(choice.measurement.asymptote, asymptote_variance(choice.measurement), choice.tree.best_choice.measurement.asymptote)
#end
#
#function prob_beat_best(reward::Float64, variance::Float64, best_reward::Float64)::Float64
#  return 1 - gumbel_cdf(reward, variance, best_reward)
#end

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

#function gumbel_cdf(mode, variance, value)
#  if variance == 0
#    if value >= mode
#      return 1  # The PDF spikes at the mode.
#    else
#      return 0
#    end
#  end
#  beta = sqrt(variance * 6 / pi^2)
#  return exp(-exp(-(value-mode)/beta))
#end
#
#function gumbel_pdf(mode, variance, value)
#  if variance == 0
#    if value == mode
#      return 1  # The PDF spikes at the mode.
#    else
#      return 0
#    end
#  end
#  beta = sqrt(variance * 6 / pi^2)
#  z = (value - mode) / beta
#  p = exp(-(z + exp(-z))) / beta
#  if p > 1
#    return 1
#  end
#  return p
#end

struct ActionEstimate
  action::Vector{UInt8}
  value::Float64
end

# Use local estimates to pick the best action from the list of uncached actions.
# Convert it to a choice.
function add_choice_from_best_uncached_action!(
           tree::Tree, guesses::Vector{Vector{UInt8}},
           solutions::Vector{Vector{UInt8}})::Union{Choice, Nothing}
  action_estimates = uncached_action_estimates(tree, guesses, solutions)
  best_action_estimate, _ = find_best_action_estimate(action_estimates)
  if isnothing(best_action_estimate)
    return nothing
  end

  choice = Choice(tree, best_action_estimate.action, solutions,
    best_action_estimate.value)
  if length(tree.choices) == 0
    tree.best_choice = choice
    tree.best_choice_lower_bound = choice
  end

  push!(tree.choices, choice)
  tree.choice_from_guess[choice.guess] = choice
  tree.newest_choice = choice

  resize_tree_stats!(tree, 1)
  tree.estimator_stats.actions_with_visits[1] += 1

  return choice
end

## Use local estimates to pick the best action from the list of uncached actions.
## Convert it to a choice.
#function add_choice_from_best_uncached_action!(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Union{Choice, Nothing}
#  action_estimates = uncached_action_estimates(tree, guesses, solutions)
#  best_action_estimate, second_best_action_estimate = find_best_action_estimate(action_estimates)
#  if isnothing(best_action_estimate)
#    return nothing
#  end
#  # We use the second variance, as the variance with a single measurement is 0.
#  uncached_variance = if length(tree.differentials.variance) < 2 || tree.differentials.variance[2] == 0
#    best_action_estimate.value^2
#  else
#    tree.differentials.variance[2]
#  end
#
#  if isnothing(second_best_action_estimate)
#    tree.prob_uncached_beat_best = 0
#  else
#    # Take the opportunity to update the prob that an uncached action surpasses
#    # the best reward across all actions given perfect play.
#    best_reward = if isnothing(tree.best_choice)
#      best_action_estimate.value
#    else
#      tree.best_choice.measurement.asymptote
#    end
#    tree.prob_uncached_beat_best = prob_beat_best(second_best_action_estimate.value, uncached_variance, best_reward)
#  end
#
#  choice = Choice(best_action_estimate.action, solutions, best_action_estimate.value, uncached_variance)
#  choice.tree = tree
#  push!(tree.choices, choice)
#  tree.choice_from_guess[best_action_estimate.action] = choice
#  return choice
#end

# Return a map from guesses to local estimates for the number of guesses to win.
function uncached_action_estimates(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Vector{ActionEstimate}
  action_estimates = Vector{ActionEstimate}()
  for guess in guesses
    if !isnothing(tree) && haskey(tree.choice_from_guess, guess)
      continue
    end
    estimated_value = estimate_guesses_remaining(guess, solutions)
    if isinf(estimated_value)
      continue
    end
    #bias = if length(tree.differentials.biases) == 0
    #  0
    #else
    #  tree.differentials.biases[1]
    #end
    #debiased_reward = estimated_value + bias
    #push!(action_estimates, ActionEstimate(guess, debiased_reward))
    push!(action_estimates, ActionEstimate(guess, estimated_value))
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
    if estimate.value > max
      max = estimate.value
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
#  choice = Choice(best_guess, solutions, best_optimal_estimate, choice_optimal_estimate_variance)
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

function Base.show(io::IO, tree::Tree)
  println(io, "Best: ", str_from_word(tree.best_choice.guess))
  #println(io, "tree.sum_prob_optimal = ", tree.sum_prob_optimal)
  #println(io, "tree.slopes = ", join(map(s -> @sprintf("%.3f", s), tree.differentials.slopes[1:min(10, length(tree.differentials.slopes))]), ", "))
  println(io, "tree.visits = ", tree.visits)
  println(io, tree.estimator_stats)
  for c in tree.choices
    println(io, c)
    #println(str_from_word(c.guess), " ", @sprintf("%.4f", -c.best_lower_bound),
    #  "~", @sprintf("%.4f", -c.reward_estimator.mean),
    #  "→", @sprintf("%.4f", -c.reward_estimator.debiased),
    #  "±", @sprintf("%.4f", sqrt(c.reward_estimator.variance)),
    #  #" o=", @sprintf("%.4f", c.prob_optimal / tree.sum_prob_optimal),
    #  #" i=", @sprintf("%.4f", c.prob_improvement),
    #  " er=", @sprintf("%.4f", c.exploratory_reward),
    #  " v=", c.visits,
    # )
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
