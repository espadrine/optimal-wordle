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

    for i in 1:1000
      improve!(tree, remaining_solutions, allowed_guesses)

      # Print best guesses so far.
      sort!(tree.choices, by = c -> c.guesses_remaining)
      choice = first(tree.choices)
      println("We suggest ", choice.guess, " (",
              @sprintf("%.5f", choice.guesses_remaining), "/",
              @sprintf("%.5f", choice.guesses_remaining_low), "/",
              @sprintf("%.5f", choice.avg_remaining), "), ",
              "step ", i)

      # Check the average number of guesses
      #println("Total guesses: ", total_guesses_for_all_sols(tree, allowed_guesses, remaining_solutions))
    end

    println("Insert your guess: ")
    guess = readline(stdin)
    println("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ")
    constraint_template = readline(stdin)
    constraint = parse_constraints(constraint_template)
    # TODO: build subtree when it does not exist.
    tree = tree.choices[findfirst(c -> c.guess == guess, tree.choices)].constraints[constraint + 1]
    remaining_solutions = filter_solutions_by_constraint(remaining_solutions, guess, constraint)
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
    @inbounds choices[i] = Choice(guess, avg_rem, nsols, nsols, 1, nothing)
  end
  for (i, choice) in enumerate(choices)
    choice.guesses_remaining_low = lower_bound_guesses_remaining(choice, solutions)
    exploratory_choices[i] = choice
  end
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
  sort!(tree.exploratory_choices, by = c -> c.guesses_remaining_low)
  choice = first(tree.exploratory_choices)
  init_guesses_remaining_low = choice.guesses_remaining_low
  init_guesses_remaining_estimate = 1.12 * (1 + log(nsolutions) / log(nsolutions / choice.avg_remaining))
  #println("Improve choice ", choice.guess, " with low ", choice.guesses_remaining_low)
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
  choice.guesses_remaining_low = lower_bound_guesses_remaining(choice, solutions)
  if nsolutions == 2315
    println("Improving from ", nsolutions, " solutions, was recommending ", choice.guess, " (",
            @sprintf("%.5f", choice.guesses_remaining), "←",
            @sprintf("%.5f", init_guesses_remaining_estimate), "/",
            @sprintf("%.5f", init_guesses_remaining_low), "/",
            @sprintf("%.5f", choice.avg_remaining), "; visits ", choice.visits, ")")
  end
  return choice.guesses_remaining
end

function lower_bound_guesses_remaining(choice::Choice, solutions::Vector{String})
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

function total_guesses_for_all_sols(tree::Tree, guesses::Vector{String}, solutions::Vector{String})::Int
  nguesses = 0
  for solution in solutions
    nguesses += total_guesses_for_sol(tree, solution, guesses, solutions)
  end
  nguesses
end

function total_guesses_for_sol(tree::Tree, solution::String, guesses::Vector{String}, solutions::Vector{String})::Int
  if length(solutions) == 1
    return 1
  end
  sort!(tree.choices, by = c -> c.guesses_remaining)
  choice = tree.choices[1]
  if choice.avg_remaining == 1
    for c in tree.choices
      if c.guess in solutions && c.avg_remaining == 1
        choice = c
      end
    end
  end
  if choice.constraints == nothing
    choice.constraints = Vector{Union{Tree, Nothing}}(nothing, 243)
  end
  c = constraints(choice.guess, solution)
  if c == 0xf2
    return 1
  end
  if choice.constraints[c + 1] == nothing
      choice.constraints[c + 1] = newTree(guesses, solutions)
  end
  remaining_solutions = filter_solutions_by_constraint(solutions, choice.guess, c)
  return 1 + total_guesses_for_sol(choice.constraints[c + 1], solution, guesses, remaining_solutions)
end

main()
