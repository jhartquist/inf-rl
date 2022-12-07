use std::cmp::min;

use crate::environment::{Environment, StepResult};

#[derive(Debug, Clone, Copy)]
pub enum GridAction {
    Up,
    Down,
    Left,
    Right,
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
}

impl FrozenLake {
    pub fn new(rows: usize, cols: usize, start: usize, goal: usize, holes: Vec<usize>) -> Self {
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
            pos: 0,
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

    fn next_position(&self, position: usize, action: &GridAction) -> usize {
        let mut row = position / self.cols;
        let mut col = position % self.cols;
        (row, col) = match action {
            GridAction::Up => (row.saturating_sub(1), col),
            GridAction::Down => (min(row + 1, self.cols - 1), col),
            GridAction::Left => (row, col.saturating_sub(1)),
            GridAction::Right => (row, min(col + 1, self.rows - 1)),
        };
        row * self.cols + col
    }

    pub fn render(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = row * self.cols + col;
                let c = if index == self.pos {
                    'A'
                } else {
                    self.grid[index].render()
                };
                print!("{}", c);
            }
            println!()
        }
        println!();
    }
}

impl Environment for FrozenLake {
    type State = usize;
    type Action = GridAction;

    fn current_state(&self) -> &Self::State {
        &self.pos
    }

    fn step(&mut self, action: &GridAction) -> Result<StepResult<usize>, String> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hole_termination() {
        let mut env = FrozenLake::new(4, 4, 0, 15, vec![5, 7, 11, 12]);

        let state = env.reset().clone();
        assert_eq!(state, 0);
        assert_eq!(state, *env.current_state());
        assert_eq!(state, env.pos);
        assert_eq!(env.is_done(), false);

        let result = env.step(&GridAction::Right).unwrap();
        assert_eq!(result.state, 1);
        assert_eq!(result.is_done, false);
        assert_eq!(result.reward, 0.0);

        env.step(&GridAction::Right).unwrap();
        env.step(&GridAction::Right).unwrap();
        assert_eq!(env.pos, 3);

        // moving toward a wall does not change the state
        env.step(&GridAction::Right).unwrap();
        assert_eq!(env.pos, 3);

        let result = env.step(&GridAction::Down).unwrap();
        assert_eq!(result.state, 7);
        assert_eq!(result.is_done, true);
        assert_eq!(result.reward, 0.0);

        let result = env.step(&GridAction::Down);
        assert!(result.is_err());
    }

    #[test]
    fn test_goal_termination() {
        let mut env = FrozenLake::new(4, 4, 0, 15, vec![5, 7, 11, 12]);
        env.reset();
        env.step(&GridAction::Down).unwrap();
        env.step(&GridAction::Down).unwrap();
        env.step(&GridAction::Right).unwrap();
        env.step(&GridAction::Right).unwrap();
        env.step(&GridAction::Down).unwrap();

        // agent reaches goal
        let result = env.step(&GridAction::Right).unwrap();
        assert!(result.is_done);
        assert_eq!(result.reward, 1.0);
        assert!(env.is_done());
    }

    // fn test_left_wall
    // fn test_right_wall
    // fn test_top_wall
    // fn test_bottom_wall
}
