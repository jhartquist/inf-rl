use std::fmt::{Debug, Display};
use std::hash::Hash;

pub type Reward = f64;

pub trait Environment {
    type State: Clone + Hash + Debug;
    type Action: Clone + Hash + Debug + Display;

    fn current_state(&self) -> &Self::State;
    fn step(&mut self, action: &Self::Action) -> Result<StepResult<Self::State>, String>;
    fn reset(&mut self) -> &Self::State;

    fn render(&self) -> String {
        unimplemented!();
    }
}

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
