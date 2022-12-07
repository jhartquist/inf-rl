use environment::Environment;
use frozen_lake::{FrozenLake, GridAction};

mod agent;
mod environment;
mod frozen_lake;
mod mdp;

fn main() -> Result<(), String> {
    let mut env = FrozenLake::new(4, 4, 0, 15, vec![5, 7, 11, 12]);
    env.render();
    env.step(&GridAction::Up)?;
    env.step(&GridAction::Down)?;
    env.step(&GridAction::Left)?;
    env.step(&GridAction::Right)?;
    Ok(())
}
