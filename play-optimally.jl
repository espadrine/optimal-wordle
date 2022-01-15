words = readlines("solutions")
non_solution_words = readlines("non-solution-guesses")
allowed_guesses = append!(words, non_solution_words)

@enum ConstraintType has_no has_but_not_at has_at
struct Constraint
  type::ConstraintType
  letter::Char
  index::Int
end

function isequal(c0::ConstraintType, c1::ConstraintType)::Bool
  c0.type == c1.type && c0.letter == c1.letter && (c0.type == has_no || c0.index == c1.index)
end

function hash(c::ConstraintType, h::UInt = 0)::UInt
  hash(c.type, h) ⊻ hash(c.letter, h ⊻ 1) ⊻ hash(c.index, h ⊻ 2)
end

function constraints(guess::String, actual::String)::Array{Constraint}
  c = Vector{Constraint}()
  actual_letters = collect(actual)
  for (i, g, a) in zip(1:5, guess, actual)
    if g == a
      push!(c, Constraint(has_at, g, i))
    elseif g ∈ actual_letters
      push!(c, Constraint(has_but_not_at, g, i))
    else
      push!(c, Constraint(has_no, g, i))
    end
  end
  c
end

# Code: . = has_no, x = has_but_not_at, o = has_at
function parse_constraints(template::String, guess::String)::Array{Constraint}
  constraints = Vector{Constraint}()
  for (i, c) in zip(1:5, template)
    if c == 'x'
      push!(constraints, Constraint(has_but_not_at, guess[i], i))
    elseif c == 'o'
      push!(constraints, Constraint(has_at, guess[i], i))
    else
      push!(constraints, Constraint(has_no, guess[i], i))
    end
  end
  constraints
end

function match_constraint(word::String, constraint::Constraint)::Bool
  letters = collect(word)
  if constraint.type == has_no
    constraint.letter ∉ letters
  elseif constraint.type == has_but_not_at
    (constraint.letter ∈ letters) && (word[constraint.index] != constraint.letter)
  elseif constraint.type == has_at
    word[constraint.index] == constraint.letter
  end
end

function match_constraints(word::String, constraints::Array{Constraint})::Bool
  for c in constraints
    if !match_constraint(word, c)
      return false
    end
  end
  return true
  #all(map(c -> match_constraint(word, c), constraints))
end

memoize_remaining_for_constraint = Dict{Constraint,Array{String}}()
function remaining_for_constraint(words::Array{String}, constraint::Constraint)::Array{String}
  global memoize_remaining_for_constraint
  if !haskey(memoize_remaining_for_constraint, constraint)
    memoize_remaining_for_constraint[constraint] = Array(filter(w -> match_constraint(w, constraint), words))
  end
  return ∩(memoize_remaining_for_constraint[constraint], words)
end

# Number of possible words left after performing a given guess against an actual
# word of the day, and receiving the corresponding constraint information.
function remaining(guess::String, actual::String, words::Array{String})::Float64
  cs = constraints(guess, actual)
  rem = 0
  for w in words
    rem += match_constraints(w, cs) ? 1 : 0
  end
  rem
  #return foldl(function(a::Float64, w::String)::Float64
  #  return a + (match_constraints(w, cs) ? 1 : 0)
  #end, words, init = 0)
  #length(reduce(∩, map(constraint -> remaining_for_constraint(words, constraint), cs)))
end

# Average number of possible words left after performing a given guess,
# and receiving the corresponding constraint information.
function avg_remaining(guess::String, words::Array{String})::Float64
  rem = 0
  for w in words
    rem += remaining(guess, w, words)
  end
  rem / length(words)
end

struct Choice
  word::String
  avg_remaining::Float64
end

function rank_guesses(guesses::Array{String}, words::Array{String})::Array{Choice}
  wt = length(guesses)
  avg_time = 0
  choices = Array{Choice}(undef, wt)
  for (wi, g) in zip(0:wt-1, guesses)
    print("\x1b[1K\x1b[GRanking guesses... ", wi, "/", wt, " words (", Int(ceil(avg_time*(wt-wi)/60)), " min left)")
    avg_time = (wi*avg_time + @elapsed choices[wi+1] = Choice(g, avg_remaining(g, words))) / (wi+1)
  end
  #time = @elapsed choices = map(function(g::String)
  #  print("\x1b[1K\x1b[GRanking guesses... ", wi, "/", wt, " words (", Int(ceil(time*(wt-wi)/60)), " min left)")
  #  wi += 1
  #  Choice(g, avg_remaining(g, words))
  #end, guesses)
  print("\x1b[1K\x1b[G")
  sort!(choices, by = c -> c.avg_remaining)
end

function main()
  global words; global allowed_guesses
  remaining_words = copy(words)  # List of words that currently fit all known constraints.
  while length(remaining_words) > 1
    best_guesses = rank_guesses(allowed_guesses, remaining_words)[1:10]
    for (i, g) in zip(1:10, best_guesses)
      println(i, ". ", g.word, " (keeps ", g.avg_remaining, " words on average)")
    end
    println("Insert your guess: ")
    guess = readline(stdin)
    println("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ")
    constraint_template = readline(stdin)
    remaining_words = filter(w -> match_constraints(w, parse_constraints(constraint_template, guess)), remaining_words)
  end
  println("Solution: ", remaining_words[1], ".")
end
main()

# $ julia guess.jl
# Ranking guesses... 1/12972 words (24950 min left)
