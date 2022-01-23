mod wordle;
mod choice;
use choice::Choice;
use std::fs;
use std::io::{self, Write};
use std::time::Duration;
use std::thread;
use std::sync::{Mutex, Arc};
use rayon::prelude::*;

const ANSI_RESET_LINE: &str = "\x1b[1K\x1b[G";

fn rank_guesses(guesses: &Vec<&str>, solutions: &Vec<&str>) -> Vec<Choice> {
    let computed_guesses = Arc::new(Mutex::new(0));
    let computed_guesses_worker = Arc::clone(&computed_guesses);
    let guesses_worker = guesses.iter().map(|w| w.to_string()).collect::<Vec<String>>();
    let solutions_worker = solutions.iter().map(|w| w.to_string()).collect::<Vec<String>>();
    let workers: thread::JoinHandle<Vec<Choice>> = thread::spawn(move ||
        guesses_worker.into_par_iter().map(|guess| {
            let choice = Choice {
                word: guess.clone(),
                avg_remaining: choice::avg_remaining(guess.as_str(), &solutions_worker.iter().map(|w| w.as_str()).collect())
            };
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
        let suggestions = if remaining_solutions.len() == solutions.len() {
            choice::root_choices()
        } else {
            rank_guesses(&allowed_guesses, &remaining_solutions)
        };
        for (i, top_choice) in suggestions.iter().take(10).enumerate() {
            println!("{}. {} (keeps {} words on average)", i, top_choice.word, top_choice.avg_remaining);
        }

        println!("Insert your guess: ");
        let mut guess = String::new();
        io::stdin().read_line(&mut guess).ok();
        println!("Insert the results (o = letter at right spot, x = wrong spot, . = not in the word): ");
        let mut constraint_template = String::new();
        io::stdin().read_line(&mut constraint_template).ok();

        // Filter remaining solutions.
        let constraints = wordle::parse_constraints(&constraint_template, &guess);
        remaining_solutions = remaining_solutions.into_iter().filter(|sol|
            wordle::match_constraints(sol, &constraints)).collect();
    }

    println!("Solution: {}.", remaining_solutions[0]);
}
