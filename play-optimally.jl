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
    for i in 1:100
      improve!(tree, remaining_solutions, allowed_guesses)

      choice = tree.best_choice
      print(ANSI_RESET_LINE, "We suggest ", str_from_word(choice.guess), " (",
              @sprintf("%.5f", choice.guesses_remaining), "/",
              @sprintf("%.5f", choice.guesses_remaining_low), "/",
              @sprintf("%.5f", choice.avg_remaining), "), ",
              "step ", i)
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
  avg_remaining::Float64
  guesses_remaining::Float64  # Including this guess, how many guesses until the win
  guesses_remaining_low::Float64  # Lower bound of uncertainty
  # Constraint that Wordle may yield as a response to this guess.
  # We link it lazily to the next choice we make in the tree search.
  visits::Int
  previous_choice::Union{Nothing, Choice}
  constraints::Union{Nothing, Vector{Any}}
end

mutable struct Tree
  choices::Vector{Choice}
  best_choice::Choice
end

function newTree(guesses::Vector{Vector{UInt8}}, solutions::Vector{Vector{UInt8}}, previous_choice::Union{Nothing, Choice})::Tree
  nguesses = length(guesses)
  nsols = length(solutions)

  if nsols == 1
    choice = Choice(solutions[1], 1, 1, 1, 1, previous_choice, nothing)
    return Tree([choice], choice)
  end

  choices = Vector{Choice}(undef, nguesses)

  # Compute the average remaining solutions for each guess:
  # it is used as the initial policy preference.
  for (i, guess) in enumerate(guesses)
    counts = zeros(Int, 243)
    for solution in solutions
      @inbounds counts[constraints(guess, solution) + 1] += 1
    end
    avg_rem = sum(abs2, counts) / nsols
    @inbounds choices[i] = Choice(guess, avg_rem, nsols, nsols, 1, previous_choice, nothing)
  end
  for choice in choices
    choice.guesses_remaining_low = lower_bound_guesses_remaining(choice, solutions)
  end
  sort!(choices, by = c -> c.guesses_remaining_low)
  #println("Computed new tree with ", nsols, " solutions. Best guess: ", str_from_word(choices[1].guess), " (", choices[1].avg_remaining, ")")

  return Tree(choices[1:100], choices[1])
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
  #println("Improving choice ", choice_breadcrumb(choice), " from ", nsolutions, " solutions.")
  init_guesses_remaining_low = choice.guesses_remaining_low
  init_guesses_remaining_estimate = 1.12 * (1 + log(nsolutions) / log(nsolutions / choice.avg_remaining))
  explored_guesses_to_win = 0  # Number of guesses to win with current, converging, policy.
  best_guesses_to_win = 0  # Actual best, to update the best score.

  for constraint in UInt8(0):UInt8(242)
    remaining_solutions = filter_solutions_by_constraint(solutions, choice.guess, constraint)
    nrsols = length(remaining_solutions)
    if nrsols == 0
      continue
    elseif nrsols == 1
      if constraint == 0xf2  # All characters are valid: we won in 1 guess.
        explored_guesses_to_win += 1
        best_guesses_to_win += 1
      else                   # The solution is found: we win on the next guess.
        explored_guesses_to_win += 2
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

    explored_guesses_remaining = improve!(subtree, remaining_solutions, guesses)
    # For each solution, we made one guess, on top of the guesses left to end the game.
    explored_guesses_to_win += (1 + explored_guesses_remaining) * nrsols
    best_guesses_to_win += (1 + subtree.best_choice.guesses_remaining) * nrsols
  end

  # To maintain the best guess, we combine the metrics we have.
  new_guesses_remaining = explored_guesses_to_win / nsolutions
  choice.guesses_remaining = best_guesses_to_win / nsolutions
  if choice.guesses_remaining < tree.best_choice.guesses_remaining
    tree.best_choice = choice
  end

  # Recursive constant:
  # tree.best_choice.guesses_remaining matches all recursive best choices.
  # We verify it through assertion testing.
  #if nsolutions == 2315
  #  total = total_guesses_for_all_sols(tree, guesses, solutions)
  #  if abs(tree.best_choice.guesses_remaining - total / nsolutions) > 0.01
  #    println("Total guesses: ", total, "; tree best choice guesses: ", tree.best_choice.guesses_remaining)
  #  end
  #end

  # For exploration, we rely on a combined metric that estimates the lower bound
  # of the number of guesses left before winning, based on our uncertainty.
  choice.visits += 1
  choice.guesses_remaining_low = lower_bound_guesses_remaining(choice, solutions)
  #if nsolutions == 2315
  #  println("Improving from ", nsolutions, " solutions, explored ", str_from_word(choice.guess), " (",
  #          @sprintf("%.5f", choice.guesses_remaining), "←",
  #          @sprintf("%.5f", init_guesses_remaining_estimate), "/",
  #          @sprintf("%.5f", init_guesses_remaining_low), "/",
  #          @sprintf("%.5f", choice.avg_remaining), "; visits ", choice.visits - 1, ")")
  #end
  return choice.guesses_remaining
end

function best_exploratory_choice(tree::Tree)::Choice
  best = tree.choices[1]
  for choice in tree.choices
    if choice.guesses_remaining_low < best.guesses_remaining_low
      best = choice
    end
  end
  return best
end

function lower_bound_guesses_remaining(choice::Choice, solutions::Vector{Vector{UInt8}})
  nsols = length(solutions)
  policy_estimate = if choice.visits > 1
    choice.guesses_remaining
  else
    prob_sol = if choice.guess in solutions
      1 / nsols  # If this pick is a winner, there are no more guesses to make.
    else
      0
    end
    # To estimate the number of remaining guesses n to win, we assume that we
    # maintain a constant ratio q of removed solutions after each guess.
    # We have s solutions currently, such that q^(n-1) = s. Thus n = 1 + log(s)÷log(q).
    # We multiply by 1.12 as that is experimentally the ratio needed to match
    # the observed number of guesses to a win.
    1.12 * (1 + (prob_sol * 0 + (1-prob_sol) * (log(nsols) / log(nsols / choice.avg_remaining))))
  end
  # We compute the weighed average of `visits + 2` measurements:
  # - an optimistic future exploration that would find the solution in 1 guess
  #   on top of the guess for this choice (for lower bound uncertainty),
  # - and either the accurate results of previous visits using the optimal policy.
  # - or the estimate from the average number of solution remaining
  #   (for when we have no accurate result),
  return (max(2, policy_estimate - 0.4) + policy_estimate * choice.visits) / (choice.visits + 1)
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
  @test constraints("erase", "melee") == parse_constraints("x...o")
  @test constraints("erase", "agree") == parse_constraints("xxx.o")
  @test constraints("erase", "widen") == parse_constraints("x....")
  @test constraints("erase", "early") == parse_constraints("oxx..")
  @test constraints("erase", "while") == parse_constraints("....o")
  @test constraints("alias", "today") == parse_constraints("...o.")
end

#test()
main()
