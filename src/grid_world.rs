use crate::environment::Reward;
use crate::mdp::Probability;
use crate::policy::Policy;
use crate::{direction::Direction, mdp::MDP};
use itertools::Itertools;
use std::cmp::min;
use std::collections::HashMap;
use std::fmt::Write;

static DIRECTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Down,
    Direction::Left,
    Direction::Right,
];

#[rustfmt::skip]
pub static FROZEN_LAKE_4X4: [&str; 4] = [
  "SFFF", 
  "FHFH", 
  "FFFH", 
  "HFFG",
];

#[rustfmt::skip]
pub static FROZEN_LAKE_8X8: [&str; 8] = [
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

    pub fn next_position(&self, position: usize, action: Direction) -> usize {
        let mut row = position / self.n_cols;
        let mut col = position % self.n_cols;
        (row, col) = match action {
            Direction::Up => (row.saturating_sub(1), col),
            Direction::Down => (min(row + 1, self.n_cols - 1), col),
            Direction::Left => (row, col.saturating_sub(1)),
            Direction::Right => (row, min(col + 1, self.n_rows - 1)),
        };
        row * self.n_cols + col
    }

    pub fn direction_probs(&self) -> HashMap<Direction, Vec<(Direction, Probability)>> {
        Direction::all()
            .into_iter()
            .map(|dir| {
                let mut dir_probs = vec![];

                // intended direction
                dir_probs.push((dir, 1.0 - self.noise));

                // noisy directions
                if self.noise > 0.0 {
                    let noisy_prob = self.noise / 2.0;
                    for noisy_dir in dir.perpindicular() {
                        dir_probs.push((noisy_dir, noisy_prob))
                    }
                }
                (dir, dir_probs)
            })
            .collect()
    }

    pub fn render_policy(&self, policy: &impl Policy<usize, Direction>) -> String {
        {
            let mut s = String::new();
            for row in 0..self.n_rows {
                for col in 0..self.n_cols {
                    let index = row * self.n_cols + col;
                    let action = policy.get_action(&index);
                    write!(s, "{} ", action).unwrap();
                }
                writeln!(s).unwrap();
            }
            writeln!(s).unwrap();
            s
        }
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

pub fn make_grid_world_mdp(grid_world: &GridWorld) -> GridWorldMDP {
    let states: Vec<usize> = (0..grid_world.grid.len()).collect();
    let actions = Direction::all();
    let rewards = grid_world.grid.iter().map(|cell| cell.reward).collect();

    let direction_probs = grid_world.direction_probs();

    let transitions = states
        .iter()
        .cartesian_product(actions.iter())
        .map(|(&state, &action)| {
            let cell = grid_world.grid[state];

            let transitions = if !cell.is_terminal {
                let mut next_state_probs = HashMap::new();
                for &(noisy_action, prob) in direction_probs[&action].iter() {
                    let next_state = grid_world.next_position(state, noisy_action);
                    *next_state_probs.entry(next_state).or_insert(0.0) += prob;
                }
                next_state_probs.into_iter().collect()
            } else {
                vec![]
            };

            ((state, action), transitions)
        })
        .collect();

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
}
