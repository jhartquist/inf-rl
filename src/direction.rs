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

    pub fn perpindicular(&self) -> [Self; 2] {
        match self {
            Direction::Up | Direction::Down => [Direction::Left, Direction::Right],
            Direction::Left | Direction::Right => [Direction::Up, Direction::Down],
        }
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let symbol = match self {
            Direction::Up => "↑",
            Direction::Down => "↓",
            Direction::Left => "←",
            Direction::Right => "→",
        };
        write!(f, "{}", symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_perpindicular() {
        assert_eq!(
            Direction::Left.perpindicular(),
            [Direction::Up, Direction::Down]
        );
        assert_eq!(
            Direction::Right.perpindicular(),
            [Direction::Up, Direction::Down]
        );
        assert_eq!(
            Direction::Up.perpindicular(),
            [Direction::Left, Direction::Right]
        );
        assert_eq!(
            Direction::Down.perpindicular(),
            [Direction::Left, Direction::Right]
        );
    }
}
