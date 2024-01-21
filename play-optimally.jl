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
        add_time(computation_timers.improve, @elapsed begin
          Threads.@threads for _ = 1:Threads.nthreads()
            improve!(tree, remaining_solutions, allowed_guesses)
          end
        end)
        step += Threads.nthreads()

        choice = tree.best_choice_lower_bound
        if should_log(TIMINGS_LOG)
          println("Times: ", string(computation_timers))
        end
        if should_log(STEP_LOG)
          println("Step ", step, ": We suggest ", choice)
        end
      end
    else
      @time for _ in 1:1000
        improve!(tree, remaining_solutions, allowed_guesses)
        step += 1

        choice = tree.best_choice
        if should_log(STEP_LOG)
          print(ANSI_RESET_LINE)
          println("Step ", step, ": We suggest ", choice)
        end
      end
    end
    println()

    println("Insert your guess: ")
    guess = Vector{UInt8}(readline(stdin))
    println("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ")
    constraint_template = readline(stdin)
    constraint = parse_constraints(constraint_template)
    # Move the state to the chosen guess.
    guess_idx = findfirst(c -> c.guess == guess, tree.choices)
    if isnothing(guess_idx)
      add_choice(tree, guess, remaining_solutions)
      guess_idx = lastindex(tree.choices)
    end
    choice = tree.choices[guess_idx]
    if isnothing(choice.constraints)
      choice.constraints = Vector{Union{Tree, Nothing}}(nothing, 243)
    end
    if isnothing(choice.constraints[constraint + 1])
      choice.constraints[constraint + 1] = Tree(allowed_guesses, remaining_solutions, choice, constraint)
    end
    tree = choice.constraints[constraint + 1]
    remaining_solutions = filter_solutions_by_constraint(remaining_solutions, guess, constraint)
    println("Remaining words: ", join(map(s -> str_from_word(s), remaining_solutions), ", "))
  end
  println("Solution: ", str_from_word(remaining_solutions[1]), ".")
end


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
  # Constraint that Wordle may yield as a response to this guess.
  # We link it lazily to the next choice we make in the tree search.
  visits::Int
  last_visit::Int  # Number of tree visits during last exploration.

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
  newest_choice::Union{Choice, Nothing}
  estimator_stats::AbstractEstimatorStats
  lock::ReentrantLock
end

function Choice(guess::Vector{UInt8})::Choice
  tree = nothing
  best_lower_bound = -1
  value = ActionValue(-1, -1, -1)
  prob_optimal = 1
  visits = 0
  last_visit = -1
  constraints = nothing
  Choice(tree, guess, best_lower_bound, value, prob_optimal, visits, last_visit, constraints)
end

function Choice(tree::Tree, guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}}, value_estimate::Float64)::Choice
  nsols = Float64(length(solutions))
  best_lower_bound = -nsols
  value = ActionValue(value_estimate, tree)
  prob_optimal = 1
  visits = 0
  last_visit = tree.visits
  constraints = nothing
  return Choice(tree, guess, best_lower_bound, value, prob_optimal, visits, last_visit, constraints)
end

# Best estimate for the action value of that choice.
function action_value(choice::Choice)::Float64
  return choice.value.debiased
end

# Best estimate for the variance of the action value estimator.
function action_value_variance(choice::Choice)::Float64
  return bias_variance_from_visit_to_end(choice.tree, choice.visits)
end

function Base.show(io::IO, choice::Choice)
  print(io, str_from_word(choice.guess), " ",
    @sprintf("%.4f", -choice.best_lower_bound), ">",
    @sprintf("%.4f", -choice.value.estimate), "→",
    @sprintf("%.4f", -choice.value.tree_estimate), "→",
    @sprintf("%.4f", -action_value(choice)), "±",
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
    lock = ReentrantLock()
    tree = Tree(previous_choice, constraint, [choice], Dict([(choice.guess, choice)]), best_choice, best_choice_lower_bound, nsols, visits, newest_choice, estimator_stats, lock)
    choice.tree = tree
    estimator_stats.tree = tree
    return tree
  end

  visits = 0
  sum_prob_optimal = 0
  newest_choice = nothing
  last_non_cache_visit = -1
  estimator_stats = EstimatorStats()
  lock = ReentrantLock()
  tree = Tree(previous_choice, constraint, [], Dict{Vector{UInt8}, Choice}(), nothing, nothing, nsols, visits, newest_choice, estimator_stats, lock)
  estimator_stats.tree = tree
  add_choice_from_best_uncached_action!(tree, guesses, solutions)
  if isnothing(tree.best_choice)
    error(string("Tree: error: no best choice found, in ", choice_breadcrumb(tree.previous_choice)))
  end
  return tree
end


# We assume that measurements follow a common statistical distribution for the
# same amount of exploration, with respect to how off they are from the
# asymptotic measurement. We estimate the distribution based on all the measured
# deltas (for a given exploration count) between the initial estimate measured,
# and the expected value of the asymptotic measurement after an infinite number
# of explorations.

mutable struct EstimatorStats <: AbstractEstimatorStats
  # Number of actions that have done at least I-1 visits (I = index).
  actions_with_visits::Vector{Int}
  # For each estimate count, we keep track of the mean bias of the estimator.
  # In other words, the average difference between an estimation with I-1
  # explorations and one with I, the latter being assumed as more precise.
  visit_bias::Vector{Float64}
  # The overall bias from one estimate count, to the latest estimate.
  bias::Vector{Float64}
  # Variance of the difference between the debiased estimate after the Ith visit
  # and the I-1 visit, times the number of samples. To allow streaming computation.
  debiased_delta_variance_t::Vector{Float64}
  # The variance of the debiased estimator after I-1 visits.
  # It represents the uncertainty of its value,
  # thus works like a mean squared error compared to the true action value.
  debiased_variance::Vector{Float64}
  tree::Union{Tree, Nothing}
end

EstimatorStats() = EstimatorStats([], [], [], [], [], nothing)

function num_actions_with_at_least(s::EstimatorStats, v::Int)::Int
  return s.actions_with_visits[v+1]
end

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

# The debiased action value estimator is the tree-based action value estimator,
# plus a visit-dependent bias computed across actions in the node.
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

function update_tree_visit_bias!(choice::Choice, old_action_value::Float64, new_action_value::Float64)
  visit_bias = choice.tree.estimator_stats.visit_bias
  v = choice.visits
  action_count = num_actions_with_at_least(choice.tree.estimator_stats, v)
  visit_bias[v] = streamed_mean(visit_bias[v], new_action_value - old_action_value, action_count)
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

  # Often, the last few values have not yet seen a delta,
  # but that does not mean that the variance is zero.
  # So we replace zero variances with the latest nonzero one.
  latest_nonzero = debiased_variance[1]
  for v = 2:length(debiased_variance)
    if debiased_variance[v] == 0.0
      debiased_variance[v] = latest_nonzero
    else
      latest_nonzero = debiased_variance[v]
    end
  end
end


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
function improve!(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})
  nsolutions = length(solutions)
  if nsolutions < 2  # The last guess finds the right solution.
    return
  end

  lock(tree.lock)
  # Select the next choice based on the optimal-converging policy
  add_time(computation_timers.select_choice, @elapsed begin
    choice, choice_idx = best_exploratory_choice!(tree, solutions, guesses)
  end)
  init_best_lower_bound = choice.best_lower_bound
  action_value_lower_bound = 0.0  # Measured best, to update the best score.
  new_action_value = 0.0

  # FIXME: speed improvement: loop through solutions if they are less numerous.
  for constraint in UInt8(0):UInt8(242)
    # State transition.
    remaining_solutions = filter_solutions_by_constraint(solutions, choice.guess, constraint)
    nrsols = length(remaining_solutions)
    prob_trans = nrsols / nsolutions
    reward = -1

    if nrsols == 0
      continue
    elseif nrsols == 1
      if constraint == 0xf2  # All characters are valid: we won in 1 guess.
        new_action_value += reward * prob_trans
        action_value_lower_bound += reward * prob_trans
      else                   # The solution is found: we win on the next guess.
        new_action_value += 2 * reward * prob_trans
        action_value_lower_bound += 2 * reward * prob_trans
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
      unlock(tree.lock)
      improve!(subtree, remaining_solutions, guesses)
      lock(tree.lock)
    end

    ## Full-depth:
    #if isnothing(subtree)        # Initialize the next move.
    #  add_time(computation_timers.new_tree, @elapsed begin
    #    subtree = Tree(guesses, remaining_solutions, choice, constraint)
    #    choice.constraints[constraint + 1] = subtree
    #  end)
    #end
    #improve!(subtree, remaining_solutions, guesses)

    # Based on the Bellman optimality equation:
    new_action_value += (reward + action_value(subtree.best_choice)) * prob_trans

    # For each solution, we made one guess, on top of the guesses left to end the game.
    action_value_lower_bound += (reward + subtree.best_choice.best_lower_bound) * prob_trans
  end

  # Update information about the current best policy.
  add_time(computation_timers.add_measurement, @elapsed begin
    update_action_value!(choice, new_action_value)
  end)
  update_best_choices!(choice, choice_idx, action_value_lower_bound)
  if should_log(DEPTH_LOG) && !isnothing(findfirst(s -> str_from_word(s) == "rover", solutions))
    println("Explored ", choice_breadcrumb(choice), ": ", nsolutions, " sols (",
            @sprintf("%.4f", -init_best_lower_bound), "→",
            @sprintf("%.4f", -choice.best_lower_bound), ") ",
            choice)
    println()
  end
  unlock(tree.lock)
end

# Pick the choice that is most valuable to explore.
function best_exploratory_choice!(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Tuple{Choice, Int}
  # Ablation study:
  # ┌─────────────┬────────────────────────────────────────┐
  # │ Lower bound │ Number of steps to find the bound      │
  # │   for best  ├──────────┬───────────┬─────────┬───────┤
  # │    choice   │ Thompson │ Hoeffding │ Laplace │  PUCT │
  # ├─────────────┼──────────┼───────────┼─────────┼───────┤
  # │     <5.0000 │        6 │       205 │     280 │   378 │
  # │     <4.0000 │       32 │       376 │     284 │   413 │
  # │      3.5532 │      413 │      6849 │    2271 │  2864 │
  # │      3.5526 │      870 │     22954 │   17735 │ 20539 │
  # └─────────────┴──────────┴───────────┴─────────┴───────┘

  choice, idx = choice_from_thompson_sampling!(tree)
  #choice, idx = choice_from_ucb_hoeffding(tree)
  #choice, idx = choice_from_ucb_laplace(tree)
  #choice, idx = choice_from_puct(tree)

  # If we pick the newest choice, we uncache a choice.
  if choice == tree.newest_choice
    add_choice_from_best_uncached_action!(tree, guesses, solutions)
  end

  if isnothing(tree.previous_choice) && should_log(ACTION_SELECTION_LOG)
    println("Exploring ", choice, " (#", idx, ")")
    if should_log(ACTION_SELECTION_TREE_LOG)
      println(tree)
    end
  end

  # The two trickiest optimal guesses, based on the last ones that are found,
  # are tarse ....x pilon (2.9880 guesses left on average),
  # and tarse ..x.x build (3.0305 guesses left on average).
  #if tree_breadcrumb(tree) == "tarse ....x"
  #  println(tree)
  #end
  return choice, idx
end

# Select a choice in proprotion to the probability that it is optimal.
function choice_from_thompson_sampling!(tree::Tree)::Tuple{Choice, Int}
  update_prob_optimal!(tree)

  # Ablation study: (entropic action value estimate)
  # ┌─────────────┬───────────────────────────────────┐
  # │ Lower bound │ Number of steps to find the bound │
  # │   for best  ├─────────────┬─────────────┬───────┤
  # │    choice   │ Probabilist │ Frequentist │  Fair │
  # ├─────────────┼─────────────┼─────────────┼───────┤
  # │     <5.0000 │           6 │           6 │    30 │
  # │     <4.0000 │          32 │          24 │    31 │
  # │      3.5532 │         413 │         315 │   513 │
  # │      3.5526 │         870 │        6586 │>13850 │
  # └─────────────┴─────────────┴─────────────┴───────┘

  # Ablation study: (averaged action value estimate)
  # ┌─────────────┬───────────────────────────────────┐
  # │ Lower bound │ Number of steps to find the bound │
  # │   for best  ├─────────────┬─────────────┬───────┤
  # │    choice   │ Probabilist │ Frequentist │  Fair │
  # ├─────────────┼─────────────┼─────────────┼───────┤
  # │     <5.0000 │          81 │          76 │   185 │
  # │     <4.0000 │          94 │          76 │   188 │
  # │      3.5532 │        3543 │        3638 │>20000 │
  # │      3.5526 │      >17000 │      >65000 │       │
  # └─────────────┴─────────────┴─────────────┴───────┘

  return probabilist_thompson_sample(tree)
  #return frequentist_thompson_sample(tree)
  #return fair_thompson_sample(tree)
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
  mode = action_value(choice)
  variance = action_value_variance(choice)
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
  # │     <4      │       32 │         35 │
  # │     <3.6    │      113 │        307 │
  # │     <3.56   │      138 │        682 │
  # │     <3.553  │      523 │       1891 │
  # │      3.5526 │      870 │      10560 │
  # └─────────────┴──────────┴────────────┘
  return sample_normal(mode, variance)
  #return sample_gumbel(mode, variance)
end

function sample_normal(mode::Float64, variance::Float64)::Float64
  if variance == 0
    return mode
  end
  return rand(Normal(mode, sqrt(variance)))
end

function sample_gumbel(mode::Float64, variance::Float64)::Float64
  if variance == 0
    return mode
  end
  scale = sqrt(variance * (6/pi^2))
  return rand(Gumbel(mode, scale))
end

# Pick choices so that the frequency that they are picked at, matches their
# probability of being the optimal action.
function probabilist_thompson_sample(tree::Tree)::Tuple{Choice, Int}
  # The set of probabilities is a pie chart,
  # and on this pie wheel, we randomly run a pointer.
  rand_pointer = rand()
  cum_prob = 0.0
  for (choice_idx, choice) in enumerate(tree.choices)
    cum_prob += choice.prob_optimal
    if rand_pointer <= cum_prob
      if isnothing(tree.previous_choice) && should_log(ACTION_SELECTION_LOG)
        println("Selection rand_pointer=", rand_pointer)
      end
      return choice, choice_idx
    end
  end
  # If a choice has not been selected yet, we pick the newest one.
  choice_idx = length(tree.choices)
  choice = tree.choices[choice_idx]
  return choice, choice_idx
end

# Pick choices so that the frequency that they are picked at, matches their
# probability of being the optimal action.
function frequentist_thompson_sample(tree::Tree)::Tuple{Choice, Int}
  # We want to pick the one with the smallest next visit.
  # The next visit is last_visit + frequency.
  idx = argmin(map(choice -> choice.last_visit + 1/choice.prob_optimal, tree.choices))
  choice = tree.choices[idx]
  return choice, idx
end

# Pick the choice based on fair total expanded work:
# a choice that has a 40% chance of being optimal
# should be explored until 40% of all historical explorations is theirs.
# We assume that the sum of prob_optimal across choices is 1.
function fair_thompson_sample(tree::Tree)::Tuple{Choice, Int}
  for (i, choice) in enumerate(tree.choices)
    if choice.visits < tree.visits * choice.prob_optimal
      return choice, i
    end
  end
  # If a choice has not been selected yet, we pick the newest one.
  idx = length(tree.choices)
  choice = tree.choices[idx]
  return choice, idx
end

function choice_from_ucb_hoeffding(tree::Tree)::Tuple{Choice, Int}
  if isnothing(tree.previous_choice) && should_log(ACTION_SELECTION_LOG)
    for choice in tree.choices
      bound = action_value_upper_bound_hoeffding(choice)
      println("Studying bound=", bound, " for ", choice)
    end
  end

  idx = argmax(map(action_value_upper_bound_hoeffding, tree.choices))
  choice = tree.choices[idx]
  return choice, idx
end

function choice_from_ucb_laplace(tree::Tree)::Tuple{Choice, Int}
  if isnothing(tree.previous_choice) && should_log(ACTION_SELECTION_LOG)
    for choice in tree.choices
      bound = action_value_upper_bound_laplace(choice)
      println("Studying bound=", bound, " for ", choice)
    end
  end

  idx = argmax(map(action_value_upper_bound_laplace, tree.choices))
  choice = tree.choices[idx]
  return choice, idx
end

function choice_from_puct(tree::Tree)::Tuple{Choice, Int}
  sum_exp_value_estimates = 0.0
  for choice in tree.choices
    sum_exp_value_estimates += exp(choice.value.estimate)
  end

  if isnothing(tree.previous_choice) && should_log(ACTION_SELECTION_LOG)
    for choice in tree.choices
      bound = action_value_upper_bound_puct(choice, sum_exp_value_estimates)
      println("Studying bound=", bound, " for ", choice)
    end
  end

  idx = argmax(map(choice -> action_value_upper_bound_puct(choice, sum_exp_value_estimates), tree.choices))
  choice = tree.choices[idx]
  return choice, idx
end

function action_value_upper_bound_hoeffding(choice::Choice)::Float64
  # Hoeffding’s inequality states that, for a sum of n indpendent random variables
  # ΣQ with L≤Qi≤U, Pr(ΣQ-𝔼[ΣQ]≥Δ) ≤ exp(-2Δ²÷(Σ(U-L)²))
  # Assuming U and L are constants, Σ(U-L)² = n(U-L)².
  # Also, since we are instead interested in the sample mean,
  # we note that ΣQ-𝔼[ΣQ]≥Δ ⇒ ΣQ÷n-𝔼[ΣQ]÷n≥Δ÷n ⇒ μ̂-μ≥Δ÷n ⇒ μ≤μ̂-Δ÷n.
  # Assuming the symmetry of the probability distribution,
  # Pr(μ≤μ̂-Δ÷n) = Pr(μ≥μ̂+Δ÷n).
  # Let’s set δ = Δ÷n, such that Δ = δ×n.
  # Then we have:
  # p := Pr(μ≥μ̂+δ) ≤ exp(-2nδ²÷(U-L)²).
  # We then have δ = (U-L)×√(-log(p)÷(2n)).
  # Experimental bounds for our action value are L=-5 and U=0.
  # We can pick p = 0.05.
  p_value = 0.999
  upper_action_value = 0
  lower_action_value = -6  # We lose the game after 6 failed guesses.
  factor = (upper_action_value - lower_action_value) * sqrt(-log(p_value)/2)
  return action_value(choice) + factor * (choice.visits+1)^-0.5
end

function action_value_upper_bound_laplace(choice::Choice)::Float64
  # Laplace’s rule of succession is typically used for discrete possibilities.
  # However, the idea can be applied to a continuous value.
  # We rely on the effect of the perturbation of another sample
  # with a fixed delta from the action value estimate.
  delta = 0.1
  return (action_value(choice) * choice.visits + action_value(choice) + delta) / (choice.visits+1)
end

function action_value_upper_bound_puct(choice::Choice, sum_exp_value_estimates::Float64)::Float64
  coeff = 1.0
  # We set the policy as the softmax of the initial action value estimate.
  policy = exp(choice.value.estimate) / sum_exp_value_estimates
  sum_visits = choice.tree.visits
  # The p_{UCT} formula is described in the AlphaGo series,
  # and while it references the Rosin 2011 paper,
  # it is widely assumed to be handcrafted with no mathematical proof
  # (cf. https://arxiv.org/pdf/2007.12509.pdf).
  # Its form varies depending on which paper you read in the AlphaGo series:
  # we use the original AlphaGo formula here:
  # https://www.rose-hulman.edu/class/cs/csse413/schedule/day16/MasteringTheGameofGo.pdf
  # AlphaGo Zero has the same formula but does not define its coefficient:
  # https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
  # AlphaZero adds 1 to its sum_visits: https://openreview.net/pdf?id=bERaNdoegnO
  # MuZero goes back to the original formula
  # but adds a visit-dependent term to the coefficient:
  # https://arxiv.org/pdf/1911.08265.pdf
  return action_value(choice) + coeff * policy * sqrt(sum_visits) / (1 + choice.visits)
end

function update_best_choices!(choice::Choice, choice_idx::Int, new_lower_bound::Float64)
  tree = choice.tree
  tree.best_choice = tree.choices[argmax(action_value(choice) for choice in tree.choices)]

  old_tree_best_lower_bound = tree.best_choice_lower_bound.best_lower_bound
  if new_lower_bound > choice.best_lower_bound
    choice.best_lower_bound = new_lower_bound
  end
  if choice.best_lower_bound > old_tree_best_lower_bound && should_log(IMPROVEMENT_LOG)
    println("Improvement found: ", choice_breadcrumb(choice), " ",
            @sprintf("%.4f", old_tree_best_lower_bound), "→",
            @sprintf("%.4f", choice.best_lower_bound),
            " (rank ", choice_idx, "; ", tree.nsolutions, " sols) ", choice)
    tree.best_choice_lower_bound = choice
  end
end

struct ActionEstimate
  action::Vector{UInt8}
  value::Float64
end

# Use local estimates to pick the best action from the list of uncached actions.
# Convert it to a choice.
function add_choice_from_best_uncached_action!(
           tree::Tree, guesses::Vector{Vector{UInt8}},
           solutions::Vector{Vector{UInt8}})::Union{Choice, Nothing}
  action_estimates = if islocked(tree.lock)
    unlock(tree.lock)
    action_estimates = uncached_action_estimates(tree, guesses, solutions)
    lock(tree.lock)
    action_estimates
  else
    uncached_action_estimates(tree, guesses, solutions)
  end
  best_action_estimate, _ = find_best_action_estimate(action_estimates)
  if isnothing(best_action_estimate)
    return nothing
  end
  return add_choice(tree::Tree, best_action_estimate.action, best_action_estimate.value, solutions::Vector{Vector{UInt8}})
end

function add_choice(tree::Tree, guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}})::Choice
  action_value = estimate_action_value(guess, solutions)
  return add_choice(tree, guess, action_value, solutions)
end

function add_choice(tree::Tree, guess::Vector{UInt8}, action_value::Float64, solutions::Vector{Vector{UInt8}})::Choice
  choice = Choice(tree, guess, solutions, action_value)
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

# Return a map from guesses to local estimates for the number of guesses to win.
function uncached_action_estimates(tree::Tree, guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}})::Vector{ActionEstimate}
  action_estimates = Vector{ActionEstimate}()
  for guess in guesses
    if !isnothing(tree) && haskey(tree.choice_from_guess, guess)
      continue
    end
    estimated_value = estimate_action_value(guess, solutions)
    if isinf(estimated_value)
      continue
    end
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

function estimate_action_value(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}})::Float64
  # Ablation study:
  # ┌─────────────┬────────────────────┐
  # │ Lower bound │ Number of steps    │
  # │   for best  ├──────────┬─────────┤
  # │    choice   │ Entropic │ Average │
  # ├─────────────┼──────────┼─────────┤
  # │     <5.0000 │        6 │      81 │
  # │     <4.0000 │       32 │      94 │
  # │      3.5532 │      413 │    3543 │
  # │      3.5526 │      870 │  >17000 │
  # └─────────────┴──────────┴─────────┘
  # Empirically, the entropic estimator is further away from the true value,
  # but its relative accuracy (between two actions) is better.
  # As a result, after a while, the variance becomes reasonable,
  # and it finds the optimal choice,
  # while the average estimator gets stuck on 3.5529.
  return -estimate_guesses_remaining_from_entropy(guess, solutions)
  #return -estimate_guesses_remaining_from_avg(guess, solutions)
end

# Including this guess, how many guesses remain until we win?
function estimate_guesses_remaining_from_entropy(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}})::Float64
  counts = zeros(Int, 243)  # Number of times the constraint appears across solutions.
  for solution in solutions
    @inbounds counts[constraints(guess, solution) + 1] += 1
  end
  nsols = length(solutions)

  entropy = 0.0
  for count in counts
    if count == 0
      continue
    end
    # Probability of a constraint appearing after we make this guess.
    prob = count / nsols
    entropy -= prob * log2(prob)
  end

  # Assuming we gain as many bits of information on each guess,
  # initial_info = info_gained_per_guess * number_of_guesses.
  # Once there are zero bits of information,
  # we know the solution for sure,
  # but we still have to submit it as a guess in order to win.
  expected_guesses = log2(nsols) / entropy + 1
  # Probability that the guess wins directly,
  # avoiding having to do another guess.
  prob_sol = if guess in solutions
    1 / nsols
  else
    0
  end
  return prob_sol * 1 + (1-prob_sol) * expected_guesses
end

# Including this guess, how many guesses remain until we win?
function estimate_guesses_remaining_from_avg(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}})::Float64
  avg_remaining = average_remaining_solutions_after_guess(guess, solutions)
  nsols = length(solutions)
  # Probability that the guess wins directly,
  # avoiding having to do another guess.
  prob_sol = if guess in solutions
    1 / nsols  # If this pick is a winner, there are no more guesses to make.
  else
    0
  end
  # To estimate the number of remaining guesses n to win, we assume that we
  # maintain a constant ratio q of removed solutions after each guess.
  # We have s solutions currently, such that q^(n-1) = s. Thus n = 1 + log(s)÷log(q).
  expected_guesses = 1 + log(nsols) / log(nsols / avg_remaining)
  return prob_sol * 1 + (1-prob_sol) * expected_guesses
end

# Compute the average remaining solutions for each guess.
function average_remaining_solutions_after_guess(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}})::Float64
  counts = zeros(Int, 243)
  for solution in solutions
    @inbounds counts[constraints(guess, solution) + 1] += 1
  end
  return sum(abs2, counts) / length(solutions)
end

function Base.show(io::IO, tree::Tree)
  println(io, "Best: ", str_from_word(tree.best_choice.guess))
  println(io, "tree.visits = ", tree.visits)
  println(io, tree.estimator_stats)
  for c in tree.choices
    println(io, c)
  end
end

function tree_breadcrumb(tree::Tree)::String
  if isnothing(tree.previous_choice)
    return "toplevel"
  end
  breadcrumb = "$(str_from_word(tree.previous_choice.guess)) $(str_from_constraints(tree.constraint))"
  tree = tree.previous_choice.tree
  while !isnothing(tree.previous_choice)
    breadcrumb = "$(str_from_word(tree.previous_choice.guess)) $(str_from_constraints(tree.constraint)) $(breadcrumb)"
    tree = tree.previous_choice.tree
  end
  return breadcrumb
end

function choice_breadcrumb(choice::Choice)
  if isnothing(choice.tree.previous_choice)
    return str_from_word(choice.guess)
  end
  return "$(tree_breadcrumb(choice.tree)) $(str_from_word(choice.guess))"
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

# Log utilities
const STEP_LOG = 1
const TIMINGS_LOG = 2
const IMPROVEMENT_LOG = 4
const ACTION_SELECTION_LOG = 8
const ACTION_SELECTION_TREE_LOG = 16
const DEPTH_LOG = 32
const ACTIVATED_LOGS = STEP_LOG | TIMINGS_LOG | IMPROVEMENT_LOG | ACTION_SELECTION_LOG
function should_log(log_level::Int)::Bool
  return (ACTIVATED_LOGS & log_level) != 0
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
