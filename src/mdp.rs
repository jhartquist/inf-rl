use std::collections::HashMap;

use crate::{
    environment::{Environment, Reward},
    policy::Policy,
};

pub trait MPD: Environment {
    fn states(&self) -> Vec<Self::State>;
    fn actions(&self) -> Vec<Self::Action>;
    fn transition(&self, state: &Self::State, action: &Self::Action) -> HashMap<Self::State, f64>;
    fn reward(
        &self,
        state: &Self::State,
        action: &Self::Action,
        next_state: &Self::State,
    ) -> Reward;

    fn render_policy<P>(&self, _policy: &P) -> String
    where
        P: Policy<Self>,
        Self: Sized,
    {
        unimplemented!();
    }
}
