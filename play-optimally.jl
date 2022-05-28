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
        improve!(tree, remaining_solutions, allowed_guesses)
        step += 1

        choice = tree.best_choice
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
  # Accuracy computation arises from the difference between the start and
  # converged measurement value.
  init_slope::Float64  # Average slope of the first half of the data points.
  # Parameters for an exponential convergence:
  # estimated_measurement(visits) = asymptote + exp_coeff × exp_base^visits
  exp_base::Float64
  exp_coeff::Float64
  # Precision of the asymptote on the first three visits.
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
  init_slope = 0
  exp_base = 0
  exp_coeff = 0
  variance = [0, 0]
  variance_visits = [0, 0]
  variance_coeff = 0
  variance_exp = 0
  return ConvergingMeasurementDifferentials(visits, max_visits, slopes, slope_visits, init_slope, exp_base, exp_coeff, variance, variance_visits, variance_coeff, variance_exp)
end

mutable struct ConvergingMeasurement
  latest::Float64
  measurements::Vector{Float64}
  visits::Int
  current::Float64  # Smoothed latest value
  average::Float64
  # Estimation of where the sequence converges.
  # Always computed directly from the previous parameters.
  asymptote::Float64
  variance::Float64  # Precision of the asymptote
  current_slope::Float64  # Average slope of the second half.
  differentials::ConvergingMeasurementDifferentials
end

function newConvergingMeasurement(differentials::ConvergingMeasurementDifferentials)::ConvergingMeasurement
  latest = 0
  measurements = [0, 0]
  visits = 0
  current = latest
  average = latest
  asymptote = latest
  variance = 0
  current_slope = 0
  return ConvergingMeasurement(latest, measurements, visits, current, average, asymptote, variance, current_slope, differentials)
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
  if aggregate.visits < 3
    aggregate.measurements[aggregate.visits] = new_measurement
  end
  aggregate.average = (aggregate.average * (aggregate.visits-1) + new_measurement) / aggregate.visits
  aggregate.current = if aggregate.visits < 2
    new_measurement
  else
    (aggregate.current + new_measurement) / 2
  end

  update_differentials!(aggregate, new_measurement - old_measurement)
  #update_exponential_params!(aggregate)
  update_asymptote!(aggregate)
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
  end
  differentials.slopes[slope_visits] = (differentials.slopes[slope_visits] * differentials.slope_visits[slope_visits] + measured_diff) / (differentials.slope_visits[slope_visits] + 1)
  differentials.slope_visits[slope_visits] += 1

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
function update_exponential_params!(aggregate::ConvergingMeasurement)
  if aggregate.visits < 3
    return  # We need at least 3 points.
  end
  differentials = aggregate.differentials

  # Slopes.
  slope_0 = 0
  slope_1 = 0
  slope_count = length(differentials.slopes)
  slope_mid = slope_count ÷ 2
  for diff in differentials.slopes[1:slope_mid]
    slope_0 += diff
  end
  for diff in differentials.slopes[slope_mid+1:slope_count]
    slope_1 += diff
  end

  # yi = a + bc^i
  # => c = (y'i ÷ y'0)^(1/i))
  # => c = ((y[n]-y[n/2]) ÷ (y[n/2]-y[0]))^(2/n))
  new_exp_base = if slope_0 == 0
    0.0
  else
    diff_quotient = slope_1 / slope_0
    if diff_quotient < 0
      0.0
    else
      diff_quotient^(2/slope_count)
    end
  end
  differentials.exp_base = new_exp_base
  if differentials.exp_base <= 0 || differentials.exp_base >= 1
    return
  end

  # => b = y'1 ÷ (c×log(c))
  new_exp_coeff = differentials.init_slope / (differentials.exp_base * log(differentials.exp_base))
  differentials.exp_coeff = new_exp_coeff
end

function update_asymptote!(aggregate::ConvergingMeasurement)
  old_asymptote = aggregate.asymptote
  aggregate.asymptote = estimate_asymptote(aggregate)
  differentials = aggregate.differentials

  for i = 1:2
    dvisits = differentials.variance_visits[i]
    squared_error = (aggregate.asymptote - aggregate.measurements[i])^2
    if aggregate.visits == i  # First visit for this aggregate: we add the squared error.
      differentials.variance_visits[i] += 1
      differentials.variance[i] = if dvisits == 0
        squared_error
      else
        differentials.variance[i] = (differentials.variance[i] * dvisits
          + squared_error) / (dvisits+1)
      end
    elseif aggregate.visits > i  # Not the first visit: we substitute the squared error.
      old_squared_error = (old_asymptote - aggregate.measurements[i])^2
      differentials.variance[i] = (differentials.variance[i] * dvisits
        - old_squared_error + squared_error) / dvisits
    end
  end

  if differentials.variance[1] > 0 && differentials.variance[2] > 0
    # y = ax^b, where a = coeff, b = exp, y = variance, x = aggregate visit.
    differentials.variance_coeff = differentials.variance[1]
    differentials.variance_exp = log2(differentials.variance[2] / differentials.variance_coeff)
  end
  aggregate.variance = estimate_variance(aggregate)
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

# Compute the bias that measurements incur by summing the averaged diffs between
# each sequential measurement.
function estimate_asymptote_from_bias(aggregate::ConvergingMeasurement)::Float64
  differentials = aggregate.differentials
  bias = 0
  v = max(aggregate.visits, 1)
  for i = v:differentials.max_visits-1
    bias += differentials.slopes[i]
  end
  return aggregate.latest + bias
end

function estimate_variance(aggregate::ConvergingMeasurement)::Float64
  if aggregate.differentials.variance_coeff == 0
    return aggregate.differentials.variance[1]
  end
  return aggregate.differentials.variance_coeff * aggregate.visits ^ aggregate.differentials.variance_exp
end

function string(aggregate::ConvergingMeasurement)::String
  differentials = aggregate.differentials
  return string("latest=", @sprintf("%.4f", aggregate.latest),
    " v=", @sprintf("%d", aggregate.visits),
    " cur=", @sprintf("%.4f", aggregate.current),
    " avg=", @sprintf("%.4f", aggregate.average),
    " asym=", @sprintf("%.4f", aggregate.asymptote),
    " var=", @sprintf("%.4f", aggregate.variance),
    " m=", @sprintf("[%.4f, %.4f]", aggregate.measurements[1], aggregate.measurements[2]),
    " cur_slope=", @sprintf("%.4f", aggregate.current_slope),
    " init_slope=", @sprintf("%.4f", differentials.init_slope),
    " exp_base=", @sprintf("%.4f", differentials.exp_base),
    " exp_coeff=", @sprintf("%.4f", differentials.exp_coeff),
    " maxv=", @sprintf("%d", differentials.max_visits),
    " dv=", @sprintf("%d", differentials.visits),
    " dvar=", @sprintf("[%.4f, %.4f]", differentials.variance[1], differentials.variance[2]),
    " dvar_coeff=", @sprintf("%.4f", differentials.variance_coeff),
    " dvar_exp=", @sprintf("%.4f", differentials.variance_exp))
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
  prob_non_cached_optimal::Float64
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
  constraints = nothing
  Choice(tree, guess, best_lower_bound, prob_optimal, prob_improvement, prob_beat_best, visits, last_visit, measurement, constraints)
end

function newChoice(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}}, optimal_estimate::Float64, optimal_estimate_variance::Float64, differentials::ConvergingMeasurementDifferentials)::Choice
  measurement = newConvergingMeasurement(differentials)
  measurement.variance = optimal_estimate_variance
  add_measurement!(measurement, optimal_estimate)

  tree = nothing
  nsols = Float64(length(solutions))
  prob_optimal = 1
  prob_improvement = 1
  prob_beat_best = 1
  visits = 0
  last_visit = -1
  constraints = nothing
  Choice(tree, guess, -nsols, prob_optimal, prob_improvement, prob_beat_best, visits, last_visit, measurement, constraints)
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
    prob_non_cached_optimal = 0
    last_non_cache_visit = -1
    tree = Tree(previous_choice, constraint, [choice], Dict([(choice.guess, choice)]), best_choice, best_choice_lower_bound, nsols, visits, sum_prob_optimal, sum_prob_beat_best, prob_non_cached_optimal, last_non_cache_visit, choice.measurement.differentials)
    choice.tree = tree
    return tree
  end

  visits = 0
  sum_prob_optimal = 0
  sum_prob_beat_best = 0
  prob_non_cached_optimal = 1
  last_non_cache_visit = -1
  tree = Tree(previous_choice, constraint, [], Dict{Vector{UInt8}, Choice}(), nothing, nothing, nsols, visits, sum_prob_optimal, sum_prob_beat_best, prob_non_cached_optimal, last_non_cache_visit, newConvergingMeasurementDifferentials())
  tree.best_choice = add_estimated_best_guess!(tree, guesses, solutions)
  tree.best_choice_lower_bound = tree.best_choice
  for choice in tree.choices
    choice.tree = tree
    choice.prob_optimal = prob_optimal_choice(choice.measurement.asymptote, choice.measurement.variance, tree)
    tree.sum_prob_optimal += choice.prob_optimal
  end
  tree.sum_prob_optimal += tree.prob_non_cached_optimal
  for choice in tree.choices
    choice.prob_beat_best = choice.prob_optimal / tree.sum_prob_optimal
    tree.sum_prob_beat_best += choice.prob_beat_best
  end
  tree.sum_prob_beat_best += tree.prob_non_cached_optimal / tree.sum_prob_optimal

  return tree
end

# We measure the mean squared error of the optimal estimate for the guess salet.
salet_accuracy = 0
salet_accuracy_count = 0

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
  choice = best_exploratory_choice_with_ordering!(tree, solutions, guesses)
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
      subtree = newTree(guesses, remaining_solutions, choice, constraint)
      choice.constraints[constraint + 1] = subtree
    else
      improve!(subtree, remaining_solutions, guesses)
    end

    # Full-depth:
    #if isnothing(subtree)        # Initialize the next move.
    #  subtree = newTree(guesses, remaining_solutions, choice, constraint)
    #  choice.constraints[constraint + 1] = subtree
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
  add_measurement!(choice, new_tree_optimal_estimate, new_guesses_remaining)
  update_prob_explore!(tree)
  println("Explored ", choice_breadcrumb(choice), ": ", nsolutions, " sols (",
          @sprintf("%.4f", -choice.best_lower_bound), "~",
          @sprintf("%.4f", -init_measurement_latest), "→",
          @sprintf("%.4f", -choice.measurement.latest), "∞→",
          @sprintf("%.4f", -choice.measurement.asymptote), "±",
          @sprintf("%.4f", sqrt(choice.measurement.variance)), ";",
          @sprintf("e=%d%%", round(init_prob_explore * 100)), "→",
          @sprintf("%d%%", round(choice.prob_beat_best / tree.sum_prob_beat_best * 100)), ";",
          @sprintf("o=%d%%", round(choice.prob_optimal / tree.sum_prob_optimal * 100)), ";",
          @sprintf("i=%d%%", round(choice.prob_improvement * 100)), ";",
          " visits ", choice.visits, ")")
  if nsolutions == 2315
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
  choice_info = best_cached_exploratory_choice(tree)
  if !isnothing(choice_info)
    return choice_info
  end
  # We had no choice whose frequency made it immediately eligible.
  # Does the non-cached choices’ probability give it an eligible frequency?
  prob_optimal_is_not_cached = tree.prob_non_cached_optimal / tree.sum_prob_optimal
  next_visit = tree.last_non_cache_visit + 1 / prob_optimal_is_not_cached
  if tree.visits >= next_visit
    return best_non_cached_exploratory_choice(tree, guesses, solutions)
  end
  return soonest_cached_exploratory_choice(tree, prob_optimal_is_not_cached, guesses, solutions)
end

function best_cached_exploratory_choice(tree::Tree)::Union{Tuple{Choice, Int}, Nothing}
  for (i, choice) in enumerate(tree.choices)
    # The frequency of visits should match the probability of exploration.
    next_visit = if choice.visits <= 0
      0.0
    else
      choice.last_visit + 1 / (choice.prob_beat_best / tree.sum_prob_beat_best)
    end
    if tree.visits >= next_visit
      if isnothing(tree.previous_choice)
        println("\nChose based on cached next visit: ", @sprintf("%.2f", next_visit))
      end
      return choice, i
    end
  end
  return nothing
end

function best_non_cached_exploratory_choice(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Tuple{Choice, Int}
  # Find the remaining guess with the best estimate.
  choice = add_estimated_best_guess!(tree, guesses, solutions)
  tree.last_non_cache_visit = tree.visits
  if isnothing(tree.previous_choice)
    println("\nChose based on non-cached frequency: ", @sprintf("%.5f", 1 / (tree.prob_non_cached_optimal / tree.sum_prob_optimal)))
  end
  return choice, length(tree.choices)
end

function soonest_cached_exploratory_choice(tree::Tree, prob_optimal_is_not_cached::Float64, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Tuple{Choice, Int}
  min_choice = tree.choices[1]
  min_idx = 1
  min_next_visit = Inf
  for (i, choice) in enumerate(tree.choices)
    next_visit = choice.last_visit + 1 / (choice.prob_beat_best / tree.sum_prob_beat_best)
    if next_visit < min_next_visit
      min_choice = choice
      min_idx = i
      min_next_visit = next_visit
    end
  end
  # The visit frequency of the next non-cached choice is taken by dividing
  # the probability that any non-cached choice is optimal, by the number of
  # non-cached choices.
  #next_non_cached_choice_visit = tree.last_non_cache_visit + 1 / (prob_optimal_is_not_cached / (length(guesses) - length(tree.choices)))
  next_non_cached_choice_visit = tree.last_non_cache_visit + 1 / prob_optimal_is_not_cached
  if next_non_cached_choice_visit < min_next_visit
    return best_non_cached_exploratory_choice(tree, guesses, solutions)
  end
  if isnothing(tree.previous_choice)
    println("\nChose based on soonest step: ", min_next_visit)
  end
  return min_choice, min_idx
end

function add_measurement!(choice::Choice, new_measurement::Float64, new_lower_bound::Float64)
  tree = choice.tree

  add_measurement!(choice.measurement, new_measurement)

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
    c.measurement.asymptote = estimate_asymptote(c.measurement)
    c.measurement.variance = estimate_variance(c.measurement)
  end
  sum_prob_optimal = 0
  for c in tree.choices
    c.prob_optimal = prob_optimal_choice(c.measurement.asymptote, c.measurement.variance, tree)
    sum_prob_optimal += c.prob_optimal
  end
  tree.sum_prob_optimal = sum_prob_optimal + tree.prob_non_cached_optimal
  sum_prob_beat_best = 0
  for c in tree.choices
    c.prob_improvement = prob_improvement(c)
    c.prob_beat_best = (c.prob_optimal / tree.sum_prob_optimal) * c.prob_improvement
    sum_prob_beat_best += c.prob_beat_best
  end
  tree.sum_prob_beat_best = sum_prob_beat_best + tree.prob_non_cached_optimal / tree.sum_prob_optimal
  if isnothing(tree.previous_choice)
    println("First choice optimal prob: ", tree.choices[1].prob_optimal)
    print_tree(tree)
  end
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
  diff_variance = optimal_estimate_variance + other.measurement.variance
  if diff_variance == 0
    return 1
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

# Probability that exploring this choice will eventually yield an improvement.
function prob_improvement(choice::Choice)::Float64
  if isnothing(choice.constraints)
    # Lacking children information, we compute the probability that the
    # asymptote is actually the lower bound or lower
    # (given the asymptote's imprecision), which would mean it no longer improves.
    return 1 - gumbel_cdf(choice.measurement.asymptote, choice.measurement.variance, choice.best_lower_bound)
  else
    # Recursive computation: it is the probability that each subtree achieves an
    # improvement, weighed by the probability that that subtree is explored.
    # - Constraint subtrees are weighted by the number of solutions, since each
    #   subtree corresponding to one constraint result, will be explored in
    #   proportion to the number of solutions that match it.
    # - Choices are weighed by prob_beat_best, as that is the exploration
    #   probability.
    sum_subtree_weights = choice.tree.nsolutions
    return foldl((p, t) -> p + if isnothing(t)
      0
    else
      subtree_weight = t.nsolutions
      sum_choice_weights = t.sum_prob_beat_best
      subtree_prob = (foldl(
        (p, c) -> p + c.prob_improvement * c.prob_beat_best,
        t.choices,
        init=0
       ) + 1 * t.prob_non_cached_optimal / t.sum_prob_optimal) / sum_choice_weights
      subtree_weight * subtree_prob
    end, choice.constraints, init=0) / sum_subtree_weights
  end
end

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

function add_estimated_best_guess!(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Choice
  nguesses = length(guesses)
  choice_estimates = Vector{Float64}(undef, nguesses)
  best_guess = guesses[1]
  best_optimal_estimate = -Inf
  optimal_estimate_mean = 0
  for (i, guess) in enumerate(guesses)
    if !isnothing(tree) && haskey(tree.choice_from_guess, guess)
      continue
    end
    optimal_estimate = estimate_guesses_remaining(guess, solutions)
    if isinf(optimal_estimate)
      continue
    end
    @inbounds choice_estimates[i] = optimal_estimate
    optimal_estimate_mean += optimal_estimate
    if optimal_estimate > best_optimal_estimate
      best_guess = guess
      best_optimal_estimate = optimal_estimate
    end
  end
  optimal_estimate_mean /= nguesses

  optimal_estimate_variance = 0
  for (i, _) in enumerate(guesses)
    @inbounds optimal_estimate_variance += (choice_estimates[i] - optimal_estimate_mean)^2
  end
  optimal_estimate_variance /= (nguesses-1)

  # The variance of the mean of a set of random variables Xi is:
  # var((ΣXi)÷N) = (Σvar(Xi))÷N² = σ²÷N
  # Thus the average variance of each estimate is this:
  choice_optimal_estimate_variance = optimal_estimate_variance * nguesses

  choice = newChoice(best_guess, solutions, best_optimal_estimate, choice_optimal_estimate_variance, tree.differentials)
  choice.tree = tree
  push!(tree.choices, choice)
  tree.choice_from_guess[best_guess] = choice

  # What is the probability that one of the non-cached choices is optimal?
  prob_non_cached_optimal = 0
  for (i, _) in enumerate(guesses)
    @inbounds prob_non_cached_optimal += prob_optimal_choice(choice_estimates[i], choice_optimal_estimate_variance, tree)
  end
  tree.prob_non_cached_optimal = prob_non_cached_optimal
  # Possible approximation:
  # 1. What is the probability that the first non-cached choice is optimal?
  #prob_non_cached_choice_is_optimal = prob_optimal_choice(second_best_optimal_estimate, choice_optimal_estimate_variance, tree)
  # 2. Assuming all non-cached choices have this probability, what is the overall probability?
  #    A choice being optimal is an exclusive event, so we can add them up.
  #tree.prob_non_cached_optimal = prob_non_cached_choice_is_optimal * (nguesses-1)

  return choice
end

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
  println("tree.prob_non_cached_optimal = ", tree.prob_non_cached_optimal)
  println("tree.differentials = ", join(map(s -> @sprintf("%.3f", s), tree.differentials.slopes[1:min(10, length(tree.differentials.slopes))]), ", "))
  for c in tree.choices
    println(str_from_word(c.guess), " ", @sprintf("%.4f", -c.best_lower_bound),
      "~", @sprintf("%.4f", -c.measurement.latest),
      "→", @sprintf("%.4f", -c.measurement.asymptote),
      "±", @sprintf("%.4f", sqrt(c.measurement.variance)),
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
