use crate::environment::Reward;
use crate::mdp::{BasicMDP, Probability};
use crate::{direction::Direction, mdp::MDP};
use itertools::Itertools;

use std::collections::HashMap;

static DIRECTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Down,
    Direction::Left,
    Direction::Right,
];

#[rustfmt::skip]
static FROZEN_LAKE_4X4: [&str; 4] = [
  "SFFF", 
  "FHFH", 
  "FFFH", 
  "HFFG",
];

#[rustfmt::skip]
static FROZEN_LAKE_8X8: [&str; 8] = [
  "SFFFFFFF",
  "FFFFFFFF",
  "FFFHFFFF",
  "FFFFFHFF",
  "FFFHFFFF",
  "FHHFFFHF",
  "FHFFHFHF",
  "FFFHFFFG",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    reward: Reward,
    is_terminal: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            reward: 0.0,
            is_terminal: false,
        }
    }
}

pub struct GridWorld {
    grid: Vec<Cell>,
    n_rows: usize,
    n_cols: usize,
    starting_states: Vec<usize>,
    noise: f64,
    discount_factor: f64,
}

impl GridWorld {
    pub fn new(
        grid: Vec<Cell>,
        n_rows: usize,
        n_cols: usize,
        starting_states: Vec<usize>,
        noise: f64,
        discount_factor: f64,
    ) -> Self {
        assert_eq!(grid.len(), n_rows * n_cols);

        GridWorld {
            grid,
            n_rows,
            n_cols,
            starting_states,
            noise,
            discount_factor,
        }
    }

    pub fn from_map(map: &[&str], noise: f64, discount_factor: f64) -> Result<Self, String> {
        let n_rows = map.len();
        if n_rows == 0 {
            return Err("zero rows".into());
        }

        let n_cols = map[0].len();
        if n_cols == 0 {
            return Err("zero cols".into());
        }

        if !map.iter().all(|s| s.len() == n_cols) {
            return Err("different length rows".into());
        }

        let mut starting_states = Vec::new();
        let grid: Vec<Cell> = map
            .into_iter()
            .flat_map(|row| row.chars())
            .enumerate()
            .map(|(i, ch)| {
                let mut cell = Cell::default();
                match ch {
                    'F' => (),
                    'S' => starting_states.push(i),
                    'H' => cell.is_terminal = true,
                    'G' => {
                        cell.reward = 1.0;
                        cell.is_terminal = true;
                    }
                    _ => return Err(format!("invalid grid cell: {}", ch)),
                }
                Ok(cell)
            })
            .collect::<Result<Vec<Cell>, String>>()?;

        Ok(GridWorld::new(
            grid,
            n_rows,
            n_cols,
            starting_states,
            noise,
            discount_factor,
        ))
    }
}

pub struct GridWorldMDP {
    states: Vec<usize>,
    transitions: HashMap<(usize, Direction), Vec<(usize, Probability)>>,
    rewards: Vec<Reward>,
}

impl GridWorldMDP {
    pub fn new(
        states: Vec<usize>,
        transitions: HashMap<(usize, Direction), Vec<(usize, Probability)>>,
        rewards: Vec<Reward>,
    ) -> Self {
        GridWorldMDP {
            states,
            transitions,
            rewards,
        }
    }
}

impl MDP for GridWorldMDP {
    type State = usize;
    type Action = Direction;

    fn get_states(&self) -> &[Self::State] {
        &self.states
    }

    fn get_actions(&self) -> &[Self::Action] {
        &DIRECTIONS
    }

    fn transition(&self, state: usize, action: Direction) -> &[(usize, Probability)] {
        &self.transitions[&(state, action)]
    }

    fn reward(&self, _state: usize, _action: Direction, next_state: usize) -> Reward {
        self.rewards[next_state]
    }
}

fn make_grid_world_mdp(grid_world: &GridWorld) -> GridWorldMDP {
    // let states: Vec<Location> = (0..grid_world.n_rows)
    //     .cartesian_product(0..grid_world.n_cols)
    //     .collect();
    let states: Vec<usize> = (0..grid_world.grid.len()).collect();
    let actions = Direction::all();
    let transitions = HashMap::new();
    let rewards = grid_world.grid.iter().map(|cell| cell.reward).collect();

    let state_actions = states.iter().cartesian_product(actions.iter());

    /*let transitions = */
    state_actions
        .map(|(&state, &action)| {
            let row = state / grid_world.n_cols;
            let col = state % grid_world.n_cols;
            println!("{} {} {} {}", state, row, col, action);
        })
        .count();

    // state_actions.into_iter().;

    // for (state, action) in states.iter().cartesian_product(actions.iter()) {
    //     println!("{:?} {}", state, action);
    // }

    GridWorldMDP::new(states, transitions, rewards)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn zero_rows() {
        let grid_world = GridWorld::from_map(&[], 0.0, 1.0);
        assert!(grid_world.is_err());
    }

    #[test]
    fn zero_cols() {
        let grid_world = GridWorld::from_map(&["", "", ""], 0.0, 1.0);
        assert!(grid_world.is_err());
    }

    #[test]
    fn different_length_rows() {
        let grid_world = GridWorld::from_map(&["FF", "FF", "F"], 0.0, 1.0);
        assert!(grid_world.is_err());
    }

    #[test]
    fn make_grid_world_4x4() {
        let grid_world = GridWorld::from_map(&FROZEN_LAKE_4X4, 0.0, 1.0).unwrap();
        assert_eq!(grid_world.grid.len(), 16);
        assert_eq!(grid_world.starting_states.len(), 1);
        assert_eq!(grid_world.grid.iter().filter(|c| c.is_terminal).count(), 5);

        let mdp = make_grid_world_mdp(&grid_world);

        let states = mdp.get_states();
        assert_eq!(states.len(), 16);

        let actions = mdp.get_actions();
        assert_eq!(actions.len(), 4);

        let total_reward: f64 = mdp.rewards.into_iter().sum();
        assert_eq!(total_reward, 1.0);
    }

    #[test]
    fn make_grid_world_8x8() {
        let grid_world = GridWorld::from_map(&FROZEN_LAKE_8X8, 0.0, 1.0).unwrap();
        let mdp = make_grid_world_mdp(&grid_world);

        let states = mdp.get_states();
        assert_eq!(states.len(), 64);

        let actions = mdp.get_actions();
        assert_eq!(actions.len(), 4);
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
