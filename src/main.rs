mod agent;
mod direction;
mod environment;
mod frozen_lake;
mod grid_world;
mod mdp;
mod policy;
mod policy_iteration;

// use frozen_lake::FrozenLake;

// use crate::mdp::MDP;

fn main() -> Result<(), String> {
    println!("hello");
    // let env = FrozenLake::new(4, 4, 0, 15, vec![5, 7, 11, 12], true);

    // let discount_rate = 0.99;
    // let threshold = 1e-10;
    // let mut rng = rand::thread_rng();

    // let policy = policy_iteration::policy_iteration(&env, discount_rate, threshold, &mut rng);
    // println!("{}", env.render_policy(&policy));

    // let policy = policy_iteration::value_iteration(&env, discount_rate, threshold);
    // println!("{}", env.render_policy(&policy));

    Ok(())
}
