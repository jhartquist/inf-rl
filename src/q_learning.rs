use std::collections::HashMap;

use ndarray::Array2;

use crate::environment::DiscreteEnvironment;
use crate::policy::DiscretePolicy;
use crate::utils::DecaySchedule;

pub fn q_learning<E>(
    env: &mut E,
    _discount_factor: f64,
    _alpha_decay: impl DecaySchedule,
    _epsilon_decay: impl DecaySchedule,
    _num_steps: usize,
    _log_interval: usize,
    _eval_episodes: usize,
) -> DiscretePolicy<E::State, E::Action>
where
    E: DiscreteEnvironment,
{
    let _q = Array2::<f64>::zeros((env.num_states(), env.num_actions()));

    DiscretePolicy::new(HashMap::new())
}

#[cfg(test)]
mod tests {
    use crate::{
        grid_world::{GridWorld, GridWorldEnv, GridWorldMDP, FROZEN_LAKE_4X4},
        utils::LinearDecay,
    };

    use super::*;

    #[test]
    fn test_frozen_lake() {
        let discount_factor = 0.99;
        let noise = 0.0;
        let grid_world = GridWorld::from_map(&FROZEN_LAKE_4X4, noise).unwrap();
        let mdp = GridWorldMDP::new(grid_world);
        let rng = rand::thread_rng();

        let mut env = GridWorldEnv::new(mdp, rng);

        let num_steps = 5000;
        let decay_alpha = LinearDecay::new(1e-2, 1e-4, num_steps);
        let decay_epsilon = LinearDecay::new(1.0, 0.1, num_steps);

        let _policy = q_learning(
            &mut env,
            discount_factor,
            decay_alpha,
            decay_epsilon,
            num_steps,
            1000,
            1000,
        );
    }
}
