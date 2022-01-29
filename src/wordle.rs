#[derive(PartialEq)]
pub enum ConstraintClass { Absent, Misplaced, Correct }

pub struct Constraint {
    class: ConstraintClass,
    letter: char,
    index: usize,
}

pub fn constraints(guess: &str, solution: &str) -> Vec<Constraint> {
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

pub fn parse_constraints(template: &str, guess: &str) -> Vec<Constraint> {
    let mut constraints = Vec::new();
    for (i, (tc, gc)) in template.chars().zip(guess.chars()).enumerate() {
        match tc {
            'o' => constraints.push(Constraint { letter: gc, index: i, class: ConstraintClass::Correct }),
            'x' => constraints.push(Constraint { letter: gc, index: i, class: ConstraintClass::Misplaced }),
            _   => (),
        }
    }
    for (i, (tc, gc)) in template.chars().zip(guess.chars()).enumerate() {
        if tc == '.' && !constraints.iter().any(|c| c.class != ConstraintClass::Absent && c.letter == gc) {
            constraints.push(Constraint { letter: gc, index: i, class: ConstraintClass::Absent });
        }
    }
    return constraints
}

pub fn match_constraint(word: &str, constraint: &Constraint) -> bool {
    match constraint.class {
        ConstraintClass::Correct => word.chars().nth(constraint.index).unwrap() == constraint.letter,
        ConstraintClass::Misplaced => word.chars().nth(constraint.index).unwrap() != constraint.letter && word.contains(constraint.letter),
        ConstraintClass::Absent => !word.contains(constraint.letter),
    }
}

pub fn match_constraints(word: &str, constraints: &Vec<Constraint>) -> bool {
    for constraint in constraints {
        if !match_constraint(word, constraint) {
            return false;
        }
    }
    return true;
}
