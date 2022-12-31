use crate::mdp::BasicMDP;
use crate::{direction::Direction, mdp::MDP};
use itertools::{Itertools, Product};

use std::collections::HashMap;

#[rustfmt::skip]
static FROZEN_LAKE_4x4: [&str; 4] = [
  "SFFF", 
  "FHFH", 
  "FFFH", 
  "HFFG",
];

#[rustfmt::skip]
static FROZEN_LAKE_8x8: [&str; 8] = [
  "SFFFFFFF",
  "FFFFFFFF",
  "FFFHFFFF",
  "FFFFFHFF",
  "FFFHFFFF",
  "FHHFFFHF",
  "FHFFHFHF",
  "FFFHFFFG",
];

fn make_grid_world_mdp(grid_map: &[&str]) -> Result<BasicMDP<'static, usize, Direction>, String> {
    let states: Vec<usize> = (0..16).collect();
    let actions = Direction::all();
    let transitions = HashMap::new();
    let rewards = HashMap::new();

    for (state, action) in states.iter().cartesian_product(actions.iter()) {}

    Ok(BasicMDP::new(states, actions, transitions, rewards))
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn make_grid_world_4x4() {
        let grid_world = make_grid_world_mdp(&FROZEN_LAKE_4x4).unwrap();

        let states = grid_world.get_states();
        assert_eq!(states.len(), 16);

        let actions = grid_world.get_actions();
        assert_eq!(actions.len(), 4);
    }

    #[test]
    fn make_grid_world_8x8() {
        let grid_world = make_grid_world_mdp(&FROZEN_LAKE_8x8).unwrap();

        let states = grid_world.get_states();
        assert_eq!(states.len(), 64);

        let actions = grid_world.get_actions();
        assert_eq!(actions.len(), 8);
    }

    #[test]
    fn test_directions() {
        let directions = Direction::all();
        assert_eq!(directions.len(), 4);

        for direction in directions {
            let opposite = direction.opposite();
            assert_ne!(direction, opposite);
            assert_eq!(direction, opposite.opposite());

            println!("{:?}", direction);
        }
    }
}
