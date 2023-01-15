use environment::Reward;
use policy::MDPPolicy;

use crate::{
    environment::Environment,
    grid_world::{GridWorldEnv, GridWorldMDP},
    policy::Policy,
};

pub mod agent;
pub mod direction;
pub mod environment;
pub mod grid_world;
pub mod mdp;
pub mod policy;
pub mod policy_iteration;

pub fn generate_episode(env: &mut GridWorldEnv, policy: &MDPPolicy<GridWorldMDP>) -> Reward {
    let mut is_done = false;
    let mut state = env.reset().clone();
    let mut total_reward = 0.0;

    while !is_done {
        let action = policy.get_action(&state);
        let result = env.step(&action).unwrap();
        is_done = result.is_done;
        state = result.state;
        total_reward += result.reward;
    }

    total_reward
}
