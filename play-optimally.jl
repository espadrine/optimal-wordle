function main()
  words = readlines("solutions")
  non_solution_words = readlines("non-solution-guesses")
  allowed_guesses = vcat(words, non_solution_words)

  remaining_words = copy(words)  # List of words that currently fit all known constraints.
  while length(remaining_words) > 1
    best_guesses = rank_guesses(allowed_guesses, remaining_words)[1:10]
    for (i, g) in zip(1:10, best_guesses)
      println(i, ". ", g.guess, " (keeps ", g.avg_remaining, " words on average)")
    end

    println("Insert your guess: ")
    guess = readline(stdin)
    println("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ")
    constraint_template = readline(stdin)
    remaining_words = filter(w -> match_constraints(w, guess, parse_constraints(constraint_template)), remaining_words)
    println("Remaining words: ", join(remaining_words, ", "))
  end
  println("Solution: ", join(remaining_words[1]), ".")
end

struct Choice
  guess::String
  avg_remaining::Float64
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
