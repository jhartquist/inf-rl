mod agent;
mod environment;
mod frozen_lake;
mod mdp;

use frozen_lake::FrozenLake;
use mdp::FiniteMDP;

fn main() -> Result<(), String> {
    let env = FrozenLake::new(4, 4, 0, 15, vec![5, 7, 11, 12], true);
    env.render();

    for state in env.states() {
        println!("{:?}", state);
        for action in env.actions() {
            println!("  {:?}", action);
            for transition in env.transition(&state, &action) {
                println!("    {:?}", transition);
            }
        }
    }

    Ok(())
}
