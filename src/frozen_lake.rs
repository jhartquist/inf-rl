use crate::{
    environment::{Environment, Reward, StepResult},
    mdp::MPD,
    policy::Policy,
};
use std::fmt::Write;
use std::{cmp::min, collections::HashMap};

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn all() -> Vec<Self> {
        vec![
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ]
    }

    pub fn opposite(&self) -> Self {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    Empty,
    Goal,
    Hole,
}

impl Cell {
    // Render the grid cell as a character.
    fn render(&self) -> char {
        match self {
            Cell::Empty => '_',
            Cell::Goal => '*',
            Cell::Hole => 'O',
        }
    }
}

pub struct FrozenLake {
    grid: Vec<Cell>,
    rows: usize,
    cols: usize,
    start: usize,
    pos: usize,
    is_slippery: bool,
}

impl FrozenLake {
    pub fn new(
        rows: usize,
        cols: usize,
        start: usize,
        goal: usize,
        holes: Vec<usize>,
        is_slippery: bool,
    ) -> Self {
        let mut grid = vec![Cell::Empty; rows * cols];

        grid[goal] = Cell::Goal;

        for hole in holes {
            grid[hole] = Cell::Hole;
        }

        Self {
            grid,
            rows,
            cols,
            start,
            pos: start,
            is_slippery,
        }
    }

    fn current_cell(&self) -> &Cell {
        &self.grid[self.pos]
    }

    fn is_done(&self) -> bool {
        match self.current_cell() {
            Cell::Hole | Cell::Goal => true,
            _ => false,
        }
    }

    fn direction_weights(&self, direction: Direction) -> Vec<(Direction, f64)> {
        if self.is_slippery {
            Direction::all()
                .into_iter()
                .filter(|&d| d != direction.opposite())
                .map(|d| (d, 1.0 / 3.0))
                .collect::<Vec<_>>()
        } else {
            vec![(direction, 1.0)]
        }
    }

    fn next_position(&self, position: usize, action: &Direction) -> usize {
        let mut row = position / self.cols;
        let mut col = position % self.cols;
        (row, col) = match action {
            Direction::Up => (row.saturating_sub(1), col),
            Direction::Down => (min(row + 1, self.cols - 1), col),
            Direction::Left => (row, col.saturating_sub(1)),
            Direction::Right => (row, min(col + 1, self.rows - 1)),
        };
        row * self.cols + col
    }
}

impl Environment for FrozenLake {
    type State = usize;
    type Action = Direction;

    fn current_state(&self) -> &Self::State {
        &self.pos
    }

    fn step(&mut self, action: &Direction) -> Result<StepResult<usize>, String> {
        if self.is_done() {
            return Err("Episode has terminated".to_string());
        }
        self.pos = self.next_position(self.pos, action);
        let reward = if self.current_cell() == &Cell::Goal {
            1.0
        } else {
            0.0
        };
        Ok(StepResult::new(self.pos, reward, self.is_done()))
    }

    fn reset(&mut self) -> &Self::State {
        self.pos = self.start;
        &self.pos
    }

    fn render(&self) -> String {
        let mut s = String::new();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = row * self.cols + col;
                let c = if index == self.pos {
                    'A'
                } else {
                    self.grid[index].render()
                };
                write!(s, "{}", c).unwrap();
            }
            writeln!(s).unwrap();
        }
        writeln!(s).unwrap();
        s
    }
}

impl MPD for FrozenLake {
    fn states(&self) -> Vec<Self::State> {
        (0..self.grid.len()).collect()
    }

    fn actions(&self) -> Vec<Self::Action> {
        Direction::all()
    }

    fn transition(&self, state: &Self::State, action: &Self::Action) -> HashMap<usize, f64> {
        let mut transitions = HashMap::new();
        for (dir, weight) in self.direction_weights(*action) {
            let next_state = self.next_position(*state, &dir);
            *transitions.entry(next_state).or_insert(0.0) += weight;
        }
        transitions
    }

    fn reward(
        &self,
        _state: &Self::State,
        _action: &Self::Action,
        next_state: &Self::State,
    ) -> Reward {
        if self.grid[*next_state] == Cell::Goal {
            1.0
        } else {
            0.0
        }
    }

    fn render_policy<P>(&self, policy: &P) -> String
    where
        P: Policy<Self>,
        Self: Sized,
    {
        {
            let mut s = String::new();
            for row in 0..self.rows {
                for col in 0..self.cols {
                    let index = row * self.cols + col;
                    let action = policy.get_action(&index);
                    write!(s, "{:?} ", action).unwrap();
                }
                writeln!(s).unwrap();
            }
            writeln!(s).unwrap();
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hole_termination() {
        let mut env = FrozenLake::new(4, 4, 0, 15, vec![5, 7, 11, 12], false);

        let state = env.reset().clone();
        assert_eq!(state, 0);
        assert_eq!(state, *env.current_state());
        assert_eq!(state, env.pos);
        assert_eq!(env.is_done(), false);

        let result = env.step(&Direction::Right).unwrap();
        assert_eq!(result.state, 1);
        assert_eq!(result.is_done, false);
        assert_eq!(result.reward, 0.0);

        env.step(&Direction::Right).unwrap();
        env.step(&Direction::Right).unwrap();
        assert_eq!(env.pos, 3);

        // moving toward a wall does not change the state
        env.step(&Direction::Right).unwrap();
        assert_eq!(env.pos, 3);

        let result = env.step(&Direction::Down).unwrap();
        assert_eq!(result.state, 7);
        assert_eq!(result.is_done, true);
        assert_eq!(result.reward, 0.0);

        let result = env.step(&Direction::Down);
        assert!(result.is_err());
    }

    #[test]
    fn test_goal_termination() {
        let mut env = FrozenLake::new(4, 4, 0, 15, vec![5, 7, 11, 12], false);
        env.reset();
        env.step(&Direction::Down).unwrap();
        env.step(&Direction::Down).unwrap();
        env.step(&Direction::Right).unwrap();
        env.step(&Direction::Right).unwrap();
        env.step(&Direction::Down).unwrap();

        // agent reaches goal
        let result = env.step(&Direction::Right).unwrap();
        assert!(result.is_done);
        assert_eq!(result.reward, 1.0);
        assert!(env.is_done());
    }

    // fn test_left_wall
    // fn test_right_wall
    // fn test_top_wall
    // fn test_bottom_wall
}
