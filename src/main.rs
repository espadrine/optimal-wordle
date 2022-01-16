use std::fs;
use std::io::{self, Write};
use std::time::Duration;
use std::thread;
use std::sync::{Mutex, Arc};
use rayon::prelude::*;

enum ConstraintClass { Absent, Misplaced, Correct }

struct Constraint {
    class: ConstraintClass,
    letter: char,
    index: usize,
}

fn constraints(guess: &str, solution: &str) -> Vec<Constraint> {
    let mut constraints = Vec::new();
    for (i, (guess_letter, solution_letter)) in guess.chars().zip(solution.chars()).enumerate() {
        constraints.push(if guess_letter == solution_letter {
            Constraint { class: ConstraintClass::Correct, letter: guess_letter, index: i }
        } else if solution.contains(guess_letter) {
            Constraint { class: ConstraintClass::Misplaced, letter: guess_letter, index: i }
        } else {
            Constraint { class: ConstraintClass::Absent, letter: guess_letter, index: i }
        });
    }
    return constraints;
}

fn parse_constraints(template: &str, guess: &str) -> Vec<Constraint> {
    let mut constraints = Vec::new();
    for (i, (tc, gc)) in template.chars().zip(guess.chars()).enumerate() {
        constraints.push(match tc {
            'o' => Constraint { letter: gc, index: i, class: ConstraintClass::Correct },
            'x' => Constraint { letter: gc, index: i, class: ConstraintClass::Misplaced },
            _   => Constraint { letter: gc, index: i, class: ConstraintClass::Absent },
        });
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

fn remaining(guess: &str, solution: &str, solutions: &Vec<String>) -> f64 {
    let constraints = constraints(guess, solution);
    let mut remaining = 0.0;
    for potential_solution in solutions {
        if match_constraints(potential_solution, &constraints) {
            remaining += 1.0;
        }
    }
    return remaining;
}

fn avg_remaining(guess: &str, solutions: &Vec<String>) -> f64 {
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

const ANSI_RESET_LINE: &str = "\x1b[1K\x1b[G";

fn rank_guesses(guesses: &Vec<&str>, solutions: &Vec<&str>) -> Vec<Choice> {
    let computed_guesses = Arc::new(Mutex::new(0));
    let computed_guesses_worker = Arc::clone(&computed_guesses);
    let guesses_worker = guesses.iter().map(|w| w.to_string()).collect::<Vec<String>>();
    let solutions_worker = solutions.iter().map(|w| w.to_string()).collect::<Vec<String>>();
    let workers: thread::JoinHandle<Vec<Choice>> = thread::spawn(move ||
        guesses_worker.par_iter().map(|guess| {
            let choice = Choice { word: guess.to_string(), avg_remaining: avg_remaining(guess, &solutions_worker) };
            let mut counter = computed_guesses_worker.lock().unwrap();
            *counter += 1;
            return choice;
        }).collect()
    );

    print!("Ranking guesses...");
    let mut guesses_per_sec = 0.0;
    let mut prev_counter = 0;
    let mut i = 0.0;
    loop {
        thread::sleep(Duration::from_secs(1));
        let counter = computed_guesses.lock().unwrap();
        if *counter >= guesses.len() { break; }
        print!("{}Ranking guesses... {}/{} words ({} min left)",
            ANSI_RESET_LINE, *counter, guesses.len(), ((guesses.len()-*counter) as f64 / guesses_per_sec / 60.0).ceil());
        io::stdout().flush().unwrap();
        guesses_per_sec = (i * guesses_per_sec + (*counter - prev_counter) as f64) / (i + 1.0);
        prev_counter = *counter;
        i += 1.0;
    }
    print!("{}", ANSI_RESET_LINE);

    let mut choices = workers.join().unwrap();
    choices.sort_by(|a, b| (*a).avg_remaining.partial_cmp(&b.avg_remaining).unwrap());
    return choices;
}

fn main() {
    // Parse valid words.
    let solutions_content = fs::read_to_string("solutions")
        .expect("Failed to read solutions file");
    let solutions: Vec<&str> = (&solutions_content).lines().collect();
    //let solutions: Vec<&str> = solutions_content.lines().take(100).collect();
    let non_solution_guesses_content = fs::read_to_string("non-solution-guesses")
        .expect("Failed to read solutions file");
    let non_solution_guesses: Vec<&str> = non_solution_guesses_content.lines().collect();
    let allowed_guesses = [&solutions[..], &non_solution_guesses[..]].concat();

    // Start the guessing game.
    let mut remaining_solutions = solutions.clone();
    while remaining_solutions.len() > 1 {
        for (i, top_choice) in rank_guesses(&allowed_guesses, &remaining_solutions).iter().take(10).enumerate() {
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
