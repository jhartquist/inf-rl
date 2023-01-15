use inf_rl::{
    generate_episode,
    grid_world::{GridWorld, GridWorldEnv, GridWorldMDP, FROZEN_LAKE_4X4, FROZEN_LAKE_8X8},
    policy_iteration,
};
use ndarray::Array;

fn main() -> Result<(), String> {
    let discount_factor = 0.99;
    let threshold = 1e-5;

    let mut rng = rand::thread_rng();
    let grid_world = GridWorld::from_map(&FROZEN_LAKE_4X4, 0.0, discount_factor).unwrap();
    let mdp = GridWorldMDP::new(grid_world);
    let policy = policy_iteration::policy_iteration(&mdp, discount_factor, threshold, &mut rng);
    let mut env = GridWorldEnv::new(mdp, rng);
    let num_episodes = 10000;
    let rewards = Array::from_iter((0..num_episodes).map(|_| generate_episode(&mut env, &policy)));
    let mean_reward = rewards.mean().unwrap();
    println!("Policy Iteration: {}", mean_reward);

    let rng = rand::thread_rng();
    let noise = 2.0 / 3.0;
    let grid_world = GridWorld::from_map(&FROZEN_LAKE_8X8, noise, discount_factor).unwrap();
    let mdp = GridWorldMDP::new(grid_world);
    let policy = policy_iteration::value_iteration(&mdp, mdp.grid_world.discount_factor, threshold);
    mdp.grid_world.render_policy(&policy);

    let mut env = GridWorldEnv::new(mdp, rng);
    let num_episodes = 10000;
    let rewards = Array::from_iter((0..num_episodes).map(|_| generate_episode(&mut env, &policy)));
    let mean_reward = rewards.mean().unwrap();
    println!("Value Iteration: {}", mean_reward);

    Ok(())
}
