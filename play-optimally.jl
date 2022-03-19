using Printf
using Test

const ANSI_RESET_LINE = "\x1b[1K\x1b[G"

function main()
  words = map(s -> Vector{UInt8}(s), readlines("solutions"))
  non_solution_words = map(s -> Vector{UInt8}(s), readlines("non-solution-guesses"))
  allowed_guesses = vcat(words, non_solution_words)

  remaining_solutions = copy(words)  # List of words that currently fit all known constraints.
  tree = newTree(allowed_guesses, remaining_solutions, nothing)
  while length(remaining_solutions) > 1
    step = 0
    if length(remaining_solutions) == 2315
      accuracy = 0
      accuracy_count = 0
      @time while tree.best_choice.best_measured_yet > 3.4212
        improve!(tree, remaining_solutions, allowed_guesses)
        step += 1

        choice = tree.best_choice
        if str_from_word(choice.guess) == "salet" && accuracy_count < 100
          accuracy = ((accuracy * accuracy_count) + (choice.optimal_estimate - 3.4212)^2) / (accuracy_count+1)
          accuracy_count += 1
        end
        print("We suggest ", str_from_word(choice.guess), " (",
                @sprintf("%.4f", choice.best_measured_yet), "~",
                @sprintf("%.4f", choice.optimal_estimate),
                "[mse=", @sprintf("%.4f", accuracy), "];",
                @sprintf("s=%d%%", round(choice.prob_beat_best * 100)), ")",
                " step ", step, ". ")
      end
    else
      @time for i in 1:1300
        improve!(tree, remaining_solutions, allowed_guesses)
        step += 1

        choice = tree.best_choice
        print(ANSI_RESET_LINE)
        print("We suggest ", str_from_word(choice.guess), " (",
                @sprintf("%.4f", choice.best_measured_yet), "~",
                @sprintf("%.4f", choice.optimal_estimate), ";",
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

mutable struct Choice
  guess::Vector{UInt8}

  # Including this guess, how many guesses until the win using the current best
  # policy. It is always an upper bound for the optimal value.
  best_measured_yet::Float64
  # Estimated slope of measurement to estimate the optimal.
  diff_best_measured_yet::Float64
  optimal_estimate::Float64
  optimal_estimate_variance::Float64

  prob_beat_best::Float64  # Probability that it can beat the best choice.
  # Constraint that Wordle may yield as a response to this guess.
  # We link it lazily to the next choice we make in the tree search.
  visits::Int
  last_visit::Int  # Number of tree visits during last exploration.
  last_improvement::Int  # Number of explorations until last improved best.

  previous_choice::Union{Nothing, Choice}
  constraints::Union{Nothing, Vector{Any}}
end

mutable struct Tree
  choices::Vector{Choice}
  best_choice::Choice
  visits::Int
  sum_prob_beat_best::Float64
  # For optimal estimate computation:
  # best_measured_yet = optimal_estimate + exp_coeff × exp_base^visits
  exp_base::Float64
  exp_coeff::Float64
end

function newChoice(guess::Vector{UInt8}, prev::Union{Nothing, Choice})::Choice
  Choice(guess, 1, 0, 1, 0, 1, 0, -1, 0, prev, nothing)
end

function newChoice(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}}, prev::Union{Nothing, Choice})::Choice
  optimal_estimate = estimate_guesses_remaining(guess, solutions)
  nsols = Float64(length(solutions))
  Choice(guess, nsols, 0, optimal_estimate, 0, 1, 0, -1, 0, prev, nothing)
end

function newTree(guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}}, previous_choice::Union{Nothing, Choice})::Tree
  nguesses = length(guesses)
  nsols = length(solutions)

  if nsols == 1
    choice = newChoice(solutions[1], previous_choice)
    return Tree([choice], choice, 0, choice.prob_beat_best, 1, 1)
  end

  choices = Vector{Choice}(undef, nguesses)
  optimal_estimate_mean = 0
  for (i, guess) in enumerate(guesses)
    @inbounds choices[i] = newChoice(guess, solutions, previous_choice)
    @inbounds optimal_estimate_mean += choices[i].optimal_estimate
  end
  optimal_estimate_mean /= length(guesses)
  sort!(choices, by = c -> c.optimal_estimate)

  optimal_estimate_variance = 0
  for choice in choices
    optimal_estimate_variance += (choice.optimal_estimate - optimal_estimate_mean)^2
  end
  optimal_estimate_variance /= (nguesses-1)

  nkept = 100
  sum_prob_beat_best = 0
  for choice in choices[1:nkept]
    choice.optimal_estimate_variance = optimal_estimate_variance / nguesses
    choice.prob_beat_best = prob_optimal_choice(choice, choices[1])
    sum_prob_beat_best += choice.prob_beat_best
  end
  #println("Computed new tree with ", nsols, " solutions. Best guess: ", str_from_word(choices[1].guess), " (", choices[1].optimal_estimate, ")")

  return Tree(choices[1:nkept], choices[1], 0, sum_prob_beat_best, 1, 1)
end

# Improve the policy by gathering data from using it with all solutions.
# Returns the average number of guesses to win across all solutions.
function improve!(tree::Tree, solutions::Vector{Vector{UInt8}}, guesses::Vector{Vector{UInt8}})::Float64
  nsolutions = length(solutions)
  nguesses = length(guesses)
  if nsolutions == 0
    return 0
  elseif nsolutions == 1
    return 1  # The last guess finds the right solution.
  end

  # Select the next choice based on the optimal-converging policy
  choice = best_exploratory_choice(tree)
  init_prob_explore = choice.prob_beat_best / tree.sum_prob_beat_best
  best_guesses_to_win = 0  # Measured best, to update the best score.

  # FIXME: speed improvement: loop through solutions if they are less numerous.
  for constraint in UInt8(0):UInt8(242)
    remaining_solutions = filter_solutions_by_constraint(solutions, choice.guess, constraint)
    nrsols = length(remaining_solutions)
    if nrsols == 0
      continue
    elseif nrsols == 1
      if constraint == 0xf2  # All characters are valid: we won in 1 guess.
        best_guesses_to_win += 1
      else                   # The solution is found: we win on the next guess.
        best_guesses_to_win += 2
      end
      continue
    end

    if choice.constraints == nothing
      choice.constraints = Vector{Union{Tree, Nothing}}(nothing, 243)
    end
    subtree = choice.constraints[constraint + 1]

    if subtree == nothing        # Initialize the next move.
      subtree = newTree(guesses, remaining_solutions, choice)
      choice.constraints[constraint + 1] = subtree
    end
    improve!(subtree, remaining_solutions, guesses)
    # For each solution, we made one guess, on top of the guesses left to end the game.
    best_guesses_to_win += (1 + subtree.best_choice.best_measured_yet) * nrsols
  end

  # Update information about the current best policy.
  new_guesses_remaining = best_guesses_to_win / nsolutions
  old_best_measured_yet = choice.best_measured_yet
  add_measurement!(choice, new_guesses_remaining, tree)
  update_optimal_estimate!(choice, old_best_measured_yet, tree)
  update_prob_explore!(choice, tree)
  if nsolutions == 2315
    println("Explored ", str_from_word(choice.guess), " (",
            @sprintf("%.4f", choice.best_measured_yet), "~",
            @sprintf("%.4f", choice.optimal_estimate), ";",
            @sprintf("e=%d%%", round(init_prob_explore * 100)), "→",
            @sprintf("%d%%", round(choice.prob_beat_best / tree.sum_prob_beat_best * 100)), ";",
            @sprintf("o=%d%%", round(prob_optimal_choice(choice, tree.best_choice) * 100)), ";",
            @sprintf("i=%d%%", round(prob_improvement(choice) * 100)), ";",
            " visits ", choice.visits, ")")
  end
  return choice.best_measured_yet
end

function best_exploratory_choice(tree::Tree)::Choice
  min_choice = tree.choices[1]
  min_next_visit = Inf
  for (i, choice) in enumerate(tree.choices)
    # The frequency of visits should match the probability of exploration.
    next_visit = if choice.last_visit < 0
      0.0
    else
      choice.last_visit + 1 / (choice.prob_beat_best / tree.sum_prob_beat_best)
    end
    if tree.visits >= next_visit
      # Constant-time iteration that converges to a sorted array.
      tree.choices[i] = tree.choices[Int(ceil(i/2))]
      tree.choices[Int(ceil(i/2))] = tree.choices[1]
      tree.choices[1] = choice
      return choice
    end
    if next_visit < min_next_visit
      min_choice = choice
      min_next_visit = next_visit
    end
  end
  return min_choice
end

function add_measurement!(choice::Choice, new_measurement::Float64, tree::Tree)
  if choice.best_measured_yet > new_measurement
    choice.best_measured_yet = new_measurement
    choice.last_improvement = choice.visits
  end
  if choice.best_measured_yet < tree.best_choice.best_measured_yet
    tree.best_choice = choice
  end
  #println("Improving choice ", choice_breadcrumb(choice), " from ", nsolutions, " solutions; best_measured_yet: ", @sprintf("%.4f", choice.best_measured_yet))

  # Recursive constant:
  # tree.best_choice.best_measured_yet matches all recursive best choices.
  # We verify it through assertion testing.
  #if nsolutions == 2315
  #  total = total_guesses_for_all_sols(tree, guesses, solutions)
  #  if abs(tree.best_choice.best_measured_yet - total / nsolutions) > 0.01
  #    println("Total guesses: ", total, "; tree best choice guesses: ", tree.best_choice.best_measured_yet)
  #  end
  #end

  # For exploration, we rely on a combined metric that estimates the lower bound
  # of the number of guesses left before winning, based on our uncertainty.
  choice.visits += 1
  choice.last_visit = tree.visits
  tree.visits += 1
end

# Improve estimate of the optimal number of guesses to win for a given choice.
function update_optimal_estimate!(choice::Choice, old_best_measured_yet::Float64, tree::Tree)
  # The theory is: the best measured guess count will exponentially improve
  # as more explorations are performed. It converges to the plateau where the
  # optimal guess count lies, with diminishing returns.
  # In order to approximate where the plateau is, we must compute estimated
  # parameters for the exponential curve.
  if choice.visits == 1
    old_best_measured_yet = choice.best_measured_yet
  end
  diff_best_measured_now = choice.best_measured_yet - old_best_measured_yet
  prev_diff_best_measured_yet = choice.diff_best_measured_yet
  # The visits count includes the current exploration.
  choice.diff_best_measured_yet = (prev_diff_best_measured_yet * choice.visits + diff_best_measured_now) / (choice.visits+1)
  if choice.diff_best_measured_yet == 0 || prev_diff_best_measured_yet == 0 || choice.diff_best_measured_yet == prev_diff_best_measured_yet
    new_optimal_estimate = choice.best_measured_yet + choice.diff_best_measured_yet
  else
    # yi = a + bc^i
    # => c = (yi-y[i-1]) ÷ (y[i-1]-y[i-2])
    new_exp_base = choice.diff_best_measured_yet / prev_diff_best_measured_yet
    tree.exp_base = (tree.exp_base * choice.visits + new_exp_base) / (choice.visits+1)
    # => b = (yi-y[i-1]) ÷ (c^i-c^(i-1))
    new_exp_coeff = choice.diff_best_measured_yet /
      (tree.exp_base^choice.visits - tree.exp_base^(choice.visits-1))
    tree.exp_coeff = (tree.exp_coeff * choice.visits + new_exp_coeff) / (choice.visits+1)
    # => a = yi - bc^i
    # FIXME: the estimate can really swing; using the average may be more stable.
    new_optimal_estimate = choice.best_measured_yet - tree.exp_coeff*tree.exp_base^choice.visits
  end
  # FIXME: Old awful estimates have a big impact on current estimates.
  # We can weigh by inverse variance.
  choice.optimal_estimate = (choice.optimal_estimate * choice.visits + new_optimal_estimate) / (choice.visits+1)
  if choice.optimal_estimate > choice.best_measured_yet
    choice.optimal_estimate = choice.best_measured_yet + choice.diff_best_measured_yet
  end
  squared_error = (new_optimal_estimate - choice.optimal_estimate)^2
  choice.optimal_estimate_variance = (choice.optimal_estimate_variance * choice.visits + squared_error) / (choice.visits+1)
end

# The likelihood that we pick a choice is its worthiness:
# the odds that it is optimal and that its exploration improves its score.
function update_prob_explore!(choice::Choice, tree::Tree)::Float64
  new_prob_beat_best = prob_optimal_choice(choice, tree.best_choice) * prob_improvement(choice)
  tree.sum_prob_beat_best = tree.sum_prob_beat_best - choice.prob_beat_best + new_prob_beat_best
  # FIXME: extreme transient values can doom good choices,
  # as their prob doesn't get recomputed until they get reconsidered
  # (which with low prob is unlikely).
  choice.prob_beat_best = new_prob_beat_best
end

# Probability that this choice is optimal under perfect play.
function prob_optimal_choice(choice::Choice, best_choice::Choice)::Float64
  variance = choice.optimal_estimate_variance + best_choice.optimal_estimate_variance
  if variance == 0
    println("prob optim ", str_from_word(choice.guess), " variance 0 for choice var=", choice.optimal_estimate_variance, " and best var=", best_choice.optimal_estimate_variance)
    return 1  # FIXME: we should have the optimal probabilities of all choices sum to 1.
  end
  #if choice.previous_choice == nothing
  #  println(" ", str_from_word(choice.guess), " prob improv: var1=", choice.optimal_estimate_variance, " var2=", best_choice.optimal_estimate_variance , " var=", variance, " p=", 1 / (1 + exp(-(0-(choice.optimal_estimate-best_choice.optimal_estimate))/(sqrt(3*variance)/pi))))
  #end
  # We now pretend the difference between this choice’s distribution
  # and the best choice’s is logistic.
  # FIXME: it is capped at 50% for the best choice;
  # comparing to a Gumbel of the other choices would be more accurate.
  1 / (1 + exp(-(0-(choice.optimal_estimate-best_choice.optimal_estimate))/(sqrt(3*variance)/pi)))
end

# Probability that exploring this choice (potentially multiple times)
# will eventually yield an improvement in its best score.
function prob_improvement(choice::Choice)::Float64
  variance = choice.optimal_estimate_variance
  if variance == 0
    println("prob impr ", str_from_word(choice.guess), " variance 0 for choice var=", choice.optimal_estimate_variance)
    return 1
  end
  # We now pretend that the optimal estimate follows a Gumbel distribution.
  # What is the probability that the optimal is lower than the best measurement
  # yet? In other words, that the best measurement has yet to improve further?
  beta = sqrt(6*variance)/pi
  mu = -choice.optimal_estimate - beta * Base.MathConstants.eulergamma
  return 1-exp(-exp((-choice.best_measured_yet-mu)/beta))
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
  # We multiply by 1.12 as that is experimentally the ratio needed to match
  # the observed number of guesses to a win.
  1.12 * (1 + (prob_sol * 0 + (1-prob_sol) * (log(nsols) / log(nsols / avg_remaining))))
end

# Compute the average remaining solutions for each guess.
function average_remaining_solutions_after_guess(guess::Vector{UInt8}, solutions::Vector{Vector{UInt8}})
  counts = zeros(Int, 243)
  for solution in solutions
    @inbounds counts[constraints(guess, solution) + 1] += 1
  end
  sum(abs2, counts) / length(solutions)
end

function lower_bound_guesses_remaining(best_measured_yet::Float64, visits::Int)
  # We compute the weighed average of `visits + 2` measurements:
  # - an optimistic future exploration that would find the solution in 1 guess
  #   on top of the guess for this choice (for lower bound uncertainty),
  # - and either the accurate results of previous visits using the optimal policy.
  # - or the estimate from the average number of solution remaining
  #   (for when we have no accurate result),
  return (max(2, best_measured_yet - 0.07) + best_measured_yet * visits) / (visits + 1)
end

function choice_breadcrumb(choice::Choice)
  choices = str_from_word(choice.guess)
  while choice.previous_choice != nothing
    choice = choice.previous_choice
    choices = @sprintf("%s %s", str_from_word(choice.guess), choices)
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
  ]

  for test in constraint_tests
    @test constraints(Vector{UInt8}(test[1]), Vector{UInt8}(test[2])) == parse_constraints(test[3])
  end
end

#test()
main()
