use crate::grid_world::{make_grid_world_mdp, GridWorld, FROZEN_LAKE_4X4, FROZEN_LAKE_8X8};

mod agent;
mod direction;
mod environment;
mod grid_world;
mod mdp;
mod policy;
mod policy_iteration;

fn main() -> Result<(), String> {
    let discount_factor = 0.99;
    let threshold = 1e-10;
    let mut rng = rand::thread_rng();

    let grid_world = GridWorld::from_map(&FROZEN_LAKE_4X4, 2.0 / 3.0, discount_factor).unwrap();
    let mdp = make_grid_world_mdp(&grid_world);

    println!("4x4\n===");
    let policy = policy_iteration::policy_iteration(&mdp, discount_factor, threshold, &mut rng);
    println!("{}", grid_world.render_policy(&policy));

    let policy = policy_iteration::value_iteration(&mdp, discount_factor, threshold);
    println!("{}", grid_world.render_policy(&policy));

    let grid_world = GridWorld::from_map(&FROZEN_LAKE_8X8, 2.0 / 3.0, discount_factor).unwrap();
    let mdp = make_grid_world_mdp(&grid_world);

    println!("8x8\n===");
    let policy = policy_iteration::policy_iteration(&mdp, discount_factor, threshold, &mut rng);
    println!("{}", grid_world.render_policy(&policy));

    let policy = policy_iteration::value_iteration(&mdp, discount_factor, threshold);
    println!("{}", grid_world.render_policy(&policy));

    Ok(())
}
