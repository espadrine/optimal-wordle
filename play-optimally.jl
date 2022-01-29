using Printf: Format, format

@enum ConstraintType has_no has_but_not_at has_at
struct Constraint
    type::ConstraintType
    letter::UInt8
    index::Int8
end

struct Choice
    word::NTuple{5,UInt8}
    avg_remaining::Float64
end

function constraints(guess, actual)::Array{Constraint}
    c = Vector{Constraint}(undef, length(guess))
    @inbounds for (i, g, a) in zip(eachindex(c), guess, actual)
        ctype = g == a ? has_at :
                g in actual ? has_but_not_at :
                has_no
        c[i] = Constraint(ctype, g, trunc(Int8, i))
    end
    return c
end

# Code: . = has_no, x = has_but_not_at, o = has_at
function parse_constraints(template, guess)::Array{Constraint}
    constraints = Vector{Constraint}(undef, length(template))
    @inbounds for (i, c) in enumerate(template)
        ctype = Char(c) == 'x' ? has_but_not_at :
                Char(c) == 'o' ? has_at :
                has_no
        constraints[i] = Constraint(ctype, UInt8(guess[i]), trunc(Int8, i))
    end
    return constraints
end

function match_constraint(word, constraint)::Bool
    if constraint.type == has_no
        constraint.letter ∉ word
    elseif constraint.type == has_but_not_at
        (constraint.letter ∈ word) && (word[constraint.index] != constraint.letter)
    elseif constraint.type == has_at
        word[constraint.index] == constraint.letter
    end
end

function match_constraints(word, constraints::Array{Constraint})::Bool
    return all(constraints) do c
        match_constraint(word, c)
    end
end

const memoize_remaining_for_constraint = Dict{Constraint,Vector{NTuple{5,UInt8}}}()

function remaining_for_constraint(words::Array{String}, constraint::Constraint)::Array{String}
    if !haskey(memoize_remaining_for_constraint, constraint)
        memoize_remaining_for_constraint[constraint] = [w for w in words if match_constraint(w, constraint)]
    end
    return ∩(memoize_remaining_for_constraint[constraint], words)
end

# Number of possible words left after performing a given guess against an actual
# word of the day, and receiving the corresponding constraint information.
function remaining(guess, actual, words::Vector)::Float64
    cs = constraints(guess, actual)
    rem = count(w -> match_constraints(w, cs), words)
    return rem
end

# Average number of possible words left after performing a given guess,
# and receiving the corresponding constraint information.
function avg_remaining(guess, words::Vector)::Float64
    rem = 0
    for w in words
        rem += remaining(guess, w, words)
    end
    rem / length(words)
end

function rank_guesses(guesses::Vector, words::Vector)
    wt = length(guesses)
    avg_time = 0
    choices = Vector{Choice}(undef, wt)
    for (wi, g) in zip(0:wt-1, guesses)
        print("\x1b[1K\x1b[GRanking guesses... $(wi)/$(wt) words ($(ceil(Int, avg_time * (wt - wi) / 60)) min left)")
        avg_time = (wi * avg_time + @elapsed choices[wi+1] = Choice(g, avg_remaining(g, words))) / (wi + 1)
    end
    print("\x1b[1K\x1b[G")
    sort!(choices, by = c -> c.avg_remaining)
end

function read_words(io::IO)
    buf = zeros(UInt8, 5)
    words = NTuple{5,UInt8}[]
    while !eof(io)
        readbytes!(io, buf)
        @inbounds word = ntuple(i -> buf[i], 5)
        push!(words, word)
        readline(io)
    end
    return words
end

read_words(fname::AbstractString) = open(read_words, fname)

function main()
    solution_file = get(ARGS, 1, "solutions")
    red_herring_file = get(ARGS, 2, "non-solution-guesses")

    words = read_words(solution_file)
    non_solution_words = read_words(red_herring_file)
    allowed_guesses = union(words, non_solution_words)

    remaining_words = words  # List of words that currently fit all known constraints.
    prop_fmt = Format("%d. %s (keeps %.4f words on average)\n")
    while length(remaining_words) > 1
        best_guesses = @view rank_guesses(allowed_guesses, remaining_words)[1:10]
        for (i, g) in zip(1:10, best_guesses)
            format(stdout, prop_fmt, i, String([g.word...]), g.avg_remaining)
        end
        println("Insert your guess: ")
        guess = readline(stdin)
        println("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ")
        constraint_template = readline(stdin)
        remaining_words = filter(w -> match_constraints(w, parse_constraints(constraint_template, guess)), remaining_words)
    end
    println("Solution: ", String([remaining_words[]...]), ".")
end

main()
