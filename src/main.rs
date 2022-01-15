use std::fs;
use std::io::{self, Write};
use std::time::Instant;

enum ConstraintClass { Absent, Misplaced, Correct }

struct Constraint {
    class: ConstraintClass,
    letter: char,
    index: usize,
}

fn constraints(guess: &str, solution: &str) -> Vec<Constraint> {
    let mut constraints = Vec::new();
    for (i, (guess_letter, solution_letter)) in guess.chars().zip(solution.chars()).enumerate() {
        if guess_letter == solution_letter {
            constraints.push(Constraint { class: ConstraintClass::Correct, letter: guess_letter, index: i });
        } else if solution.contains(guess_letter) {
            constraints.push(Constraint { class: ConstraintClass::Misplaced, letter: guess_letter, index: i });
        } else {
            constraints.push(Constraint { class: ConstraintClass::Absent, letter: guess_letter, index: i });
        }
    }
    return constraints;
}

fn parse_constraints(template: &str, guess: &str) -> Vec<Constraint> {
    let mut constraints = Vec::new();
    for (i, (tc, gc)) in template.chars().zip(guess.chars()).enumerate() {
        match tc {
            'o' => constraints.push(Constraint { letter: gc, index: i, class: ConstraintClass::Correct }),
            'x' => constraints.push(Constraint { letter: gc, index: i, class: ConstraintClass::Misplaced }),
            _   => constraints.push(Constraint { letter: gc, index: i, class: ConstraintClass::Absent }),
        }
    }
    return constraints
}

fn match_constraint(word: &str, constraint: &Constraint) -> bool {
    match constraint.class {
        ConstraintClass::Correct => word.chars().nth(constraint.index).unwrap() == constraint.letter,
        ConstraintClass::Misplaced => word.chars().nth(constraint.index).unwrap() != constraint.letter && word.contains(constraint.letter),
        ConstraintClass::Absent => !word.contains(constraint.letter),
    }
}

fn match_constraints(word: &str, constraints: &Vec<Constraint>) -> bool {
    for constraint in constraints {
        if !match_constraint(word, constraint) {
            return false;
        }
    }
    return true;
}

fn remaining(guess: &str, solution: &str, solutions: &Vec<&str>) -> f64 {
    let constraints = constraints(guess, solution);
    let mut remaining = 0.0;
    for potential_solution in solutions {
        if match_constraints(potential_solution, &constraints) {
            remaining += 1.0;
        }
    }
    return remaining;
}

fn avg_remaining(guess: &str, solutions: &Vec<&str>) -> f64 {
    let mut remaining_sols = 0.0;
    for solution in solutions {
        remaining_sols += remaining(guess, solution, &solutions);
    }
    return remaining_sols / solutions.len() as f64;
}

struct Choice {
    word: String,
    avg_remaining: f64,
}

fn rank_guesses(guesses: &Vec<&str>, solutions: &Vec<&str>) -> Vec<Choice> {
    let mut choices = Vec::new();
    let mut avg_time = 0.0;
    for (i, guess) in guesses.iter().enumerate() {
        print!("\x1b[1K\x1b[GRanking guesses... {}/{} words ({} min left)",
            i, guesses.len(), (avg_time * (guesses.len()-i) as f64 / 60.0).ceil());
        io::stdout().flush().unwrap();
        let time = Instant::now();
        choices.push(Choice { word: guess.to_string(), avg_remaining: avg_remaining(guess, &solutions) });
        avg_time = (i as f64 * avg_time + time.elapsed().as_secs_f64()) / (i+1) as f64;
    }
    print!("\x1b[1K\x1b[G");
    choices.sort_by(|a, b| a.avg_remaining.partial_cmp(&b.avg_remaining).unwrap());
    return choices;
}

fn main() {
    // Parse valid words.
    let solutions_content = fs::read_to_string("solutions")
        .expect("Failed to read solutions file");
    let solutions: Vec<&str> = solutions_content.lines().collect();
    //let solutions: Vec<&str> = solutions_content.lines().take(10).collect();
    let non_solution_guesses_content = fs::read_to_string("non-solution-guesses")
        .expect("Failed to read solutions file");
    let non_solution_guesses: Vec<&str> = non_solution_guesses_content.lines().collect();
    let allowed_guesses = [&solutions[..], &non_solution_guesses[..]].concat();

    // Start the guessing game.
    let mut remaining_solutions = solutions.clone();
    while remaining_solutions.len() > 1 {
        for (i, top_choice) in rank_guesses(&allowed_guesses, &remaining_solutions).iter().enumerate() {
            println!("{}. {} (keeps {} words on average)", i, top_choice.word, top_choice.avg_remaining);
        }

        println!("Insert your guess: ");
        let mut guess = String::new();
        io::stdin().read_line(&mut guess).ok();
        println!("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ");
        let mut constraint_template = String::new();
        io::stdin().read_line(&mut constraint_template).ok();

        // Filter remaining solutions.
        let constraints = parse_constraints(&constraint_template, &guess);
        remaining_solutions = remaining_solutions.into_iter().filter(|sol|
            match_constraints(sol, &constraints)).collect();
    }

    println!("Solution: {}.", remaining_solutions[0]);
}
