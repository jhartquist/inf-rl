use std::fmt::Debug;
use std::hash::Hash;

pub trait Environment {
    type State: Copy + Clone + Hash + Eq + Debug;
    type Action: Copy + Clone + Hash + Eq + Debug;

    fn current_state(&self) -> &Self::State;
    fn step(&mut self, action: &Self::Action) -> Result<StepResult<Self::State>, String>;
    fn reset(&mut self) -> &Self::State;
}

pub type Reward = f64;

#[derive(Debug, Clone, Copy)]
pub struct StepResult<State> {
    pub state: State,
    pub reward: Reward,
    pub is_done: bool,
}

impl<State> StepResult<State> {
    pub fn new(state: State, reward: Reward, is_done: bool) -> Self {
        Self {
            state,
            reward,
            is_done,
        }
    }
}
