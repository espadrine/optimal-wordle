using Printf

function main()
  words = readlines("solutions")
  non_solution_words = readlines("non-solution-guesses")
  allowed_guesses = vcat(words, non_solution_words)

  remaining_solutions = copy(words)  # List of words that currently fit all known constraints.
  tree = newTree(allowed_guesses, remaining_solutions)
  while length(remaining_solutions) > 1
    #best_guesses = rank_guesses(allowed_guesses, remaining_solutions)[1:10]
    #for (i, g) in zip(1:10, best_guesses)
    #  println(i, ". ", g.guess, " (keeps ", g.avg_remaining, " words on average)")
    #end

    for guess in allowed_guesses
      improve!(tree, remaining_solutions, allowed_guesses)

      # Print best guesses so far.
      choice = first(tree.choices)
      println("We suggest ", choice.guess, " (keeps ", choice.avg_remaining, " words on average; ",
              "solves in ", choice.guesses_remaining, " guesses.)")
    end

    println("Insert your guess: ")
    guess = readline(stdin)
    println("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ")
    constraint_template = readline(stdin)
    remaining_solutions = filter_solutions_by_constraint(remaining_solutions, guess, parse_constraints(constraint_template))
    println("Remaining words: ", join(remaining_solutions, ", "))
  end
  println("Solution: ", join(remaining_solutions[1]), ".")
end

struct Tree
  # The top choice has the best converging policy.
  choices::Vector{Any}
  # The top exploratory choice has the best converging lower bound policy.
  exploratory_choices::Vector{Any}
end

mutable struct Choice
  guess::String
  avg_remaining::Float64
  guesses_remaining::Float64  # Including this guess, how many guesses until the win
  guesses_remaining_low::Float64  # Lower bound of uncertainty
  # Constraint that Wordle may yield as a response to this guess.
  # We link it lazily to the next choice we make in the tree search.
  visits::Int
  constraints::Union{Nothing, Vector{Union{Tree, Nothing}}}
end

function newTree(guesses::Vector{String}, solutions::Vector{String})::Tree
  nguesses = length(guesses)
  nsols = length(solutions)

  if nsols == 1
    choice = Choice(solutions[1], 1, 1, 1, 1, nothing)
    return Tree([choice], [choice])
  end

  choices = Vector{Choice}(undef, nguesses)
  exploratory_choices = Vector{Choice}(undef, nguesses)

  # Compute the average remaining solutions for each guess:
  # it is used as the initial policy preference.
  for (i, guess) in enumerate(guesses)
    counts = zeros(Int, 243)
    for solution in solutions
      @inbounds counts[constraints(guess, solution) + 1] += 1
    end
    avg_rem = sum(abs2, counts) / nsols
    if avg_rem <= 1 && guess in solutions
      guesses_left = (1 + 2 * (nsols-1)) / nsols
      @inbounds choices[i] = Choice(guess, avg_rem, guesses_left, guesses_left, 1, nothing)
    else
      @inbounds choices[i] = Choice(guess, avg_rem, Inf, Inf, 0, nothing)
    end
  end
  for (i, choice) in enumerate(choices)
    choice.guesses_remaining_low = lower_bound_guesses_remaining(choice, nsols)
    exploratory_choices[i] = choice
  end
  sort!(choices, by = c -> c.avg_remaining)
  sort!(exploratory_choices, by = c -> c.guesses_remaining_low)
  #println("Computed new tree with ", nsols, " solutions. Best guess: ", choices[1].guess, " (", choices[1].avg_remaining, "); best exploratory guess: ", exploratory_choices[1].guess, " (", exploratory_choices[1].guesses_remaining_low, ")")

  return Tree(choices[1:100], exploratory_choices[1:100])
end

# Improve the policy by gathering data from using it with all solutions.
# Returns the average number of guesses to win across all solutions.
function improve!(tree::Tree, solutions::Vector{String}, guesses::Vector{String})::Float64
  nsolutions = length(solutions)
  nguesses = length(guesses)
  if nsolutions == 0
    return 0
  elseif nsolutions == 1
    return 1  # The last guess finds the right solution.
  end

  # Select the next choice based on the optimal-converging policy
  # (choices have already been sorted according to it).
  choice = first(tree.exploratory_choices)
  guesses_to_win = 0  # Number of guesses to win with current, converging, policy.

  for constraint in UInt8(0):UInt8(242)
    remaining_solutions = filter_solutions_by_constraint(solutions, choice.guess, constraint)
    nrsols = length(remaining_solutions)
    if nrsols == 0
      continue
    elseif nrsols == 1
      if constraint == 0xf2  # All characters are valid: we won in 1 guess.
        guesses_to_win += 1
      else                   # The solution is found: we win on the next guess.
        guesses_to_win += 2
      end
      continue
    end

    if choice.constraints == nothing
      choice.constraints = Vector{Union{Tree, Nothing}}(nothing, 243)
    end
    subtree = choice.constraints[constraint + 1]

    if subtree == nothing        # Initialize the next move.
      subtree = newTree(guesses, remaining_solutions)
      choice.constraints[constraint + 1] = subtree
    end

    avg_guesses_to_win = improve!(subtree, remaining_solutions, guesses)
    # For each solution, we made one guess, on top of the guesses left to end the game.
    guesses_to_win += (1 + avg_guesses_to_win) * nrsols
  end

  # To sort guesses, we combine the metrics we have.
  new_guesses_remaining = guesses_to_win / nsolutions
  if new_guesses_remaining < choice.guesses_remaining
    choice.guesses_remaining = new_guesses_remaining
  end
  # For exploration, we rely on a combined metric that estimates the lower bound
  # of the number of guesses left before winning, based on our uncertainty.
  choice.visits += 1
  choice.guesses_remaining_low = lower_bound_guesses_remaining(choice, nsolutions)
  sort!(tree.choices, by = c -> c.guesses_remaining)
  sort!(tree.exploratory_choices, by = c -> c.guesses_remaining_low)
  if nsolutions == 2315
    println("Improving from ", nsolutions, " solutions, was recommending ", choice.guess, " (", @sprintf("%.5f", choice.guesses_remaining), "/", @sprintf("%.5f", choice.guesses_remaining_low), "/", @sprintf("%.5f", choice.avg_remaining), "; visits ", choice.visits, "), now recommending ", tree.choices[1].guess, " (", @sprintf("%.5f", tree.choices[1].guesses_remaining), "/", @sprintf("%.5f", tree.choices[1].guesses_remaining_low), "/", @sprintf("%.5f", tree.choices[1].avg_remaining), ")")
  end
  return choice.guesses_remaining
end

function lower_bound_guesses_remaining(choice::Choice, nsols::Int)
  policy_estimate = if choice.guesses_remaining != Inf
    choice.guesses_remaining * choice.visits
  else
    0
  end
  # To estimate the number of remaining guesses n to win, we assume that we
  # maintain a constant ratio q of removed solutions after each guess.
  # We have s solutions currently, such that q^n = s. Thus n = log(s)÷log(q).
  est_guesses_rem_from_avg_rem = log(nsols) / log(nsols / choice.avg_remaining)
  # We compute the weighed average of `visits + 2` measurements:
  # - an optimistic future exploration that would find the solution in 1 guess
  #   (for lower bound uncertainty),
  # - the estimate from the average number of solution remaining
  #   (for when we have no accurate result),
  # - and the accurate results of previous visits using the optimal policy.
  return (1 + est_guesses_rem_from_avg_rem + policy_estimate) / (choice.visits + 2)
end

function rank_guesses(guesses::Array{String}, words::Array{String})::Array{Choice}
  nguesses = length(guesses)
  nwords = length(words)
  counts = zeros(Int, 243)
  choices = Vector{Choice}(undef, nguesses)
  for (i, guess) in enumerate(guesses)
    count_constraints!(counts, guess, words)
    @inbounds choices[i] = Choice(guess, sum(abs2, counts) / nwords)
  end
  sort!(choices, by = c -> c.avg_remaining)
end

function count_constraints!(counts::Array{Int}, guess::String, words::Array{String})
  fill!(counts, 0)
  for actual in words
    counts[constraints(guess, actual) + 1] += 1
  end
end

function constraints(guess, actual)::UInt8
  constraints = UInt8(0)
  mult = 1
  for (g, a) in zip(guess, actual)
    constraints += (g == a ? 2 : (g ∈ actual ? 1 : 0)) * mult
    mult *= 3
  end
  return constraints
end

function filter_solutions_by_constraint(solutions::Vector{String}, guess::String, constraint::UInt8)::Vector{String}
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

function match_constraints(word::String, guess::String, constraints::UInt8)::Bool
  for (w, g) in zip(word, guess)
    if constraints % 3 == 2
      if w != g
        return false
      end
    elseif constraints % 3 == 1
      if w == g || g ∉ word
        return false
      end
    else
      if g ∈ word
        return false
      end
    end
    constraints ÷= 3
  end
  return true
end

main()
