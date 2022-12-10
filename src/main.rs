mod agent;
mod environment;
mod frozen_lake;
mod mdp;
mod policy;
mod policy_iteration;

use frozen_lake::FrozenLake;

fn main() -> Result<(), String> {
    let env = FrozenLake::new(4, 4, 0, 15, vec![5, 7, 11, 12], true);

    policy_iteration::solve(&env);

    Ok(())
}
