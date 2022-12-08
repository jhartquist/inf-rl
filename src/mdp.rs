use std::collections::HashMap;

use crate::environment::Reward;

pub trait FiniteMDP {
    type State;
    type Action;

    fn states(&self) -> Vec<Self::State>;
    fn actions(&self) -> Vec<Self::Action>;
    fn transition(&self, state: &Self::State, action: &Self::Action) -> HashMap<Self::State, f64>;
    fn reward(
        &self,
        state: &Self::State,
        action: &Self::Action,
        next_state: &Self::State,
    ) -> Reward;
}
