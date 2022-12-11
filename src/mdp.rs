use std::collections::HashMap;

use crate::{
    environment::{Environment, Reward},
    policy::Policy,
};

pub trait MDP: Environment {
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

    fn print_transitions(&self) {
        let actions = self.actions();
        for state in self.states() {
            println!("state {:>2?}", state);
            for action in &actions {
                println!("  {}", action);
                for (next_state, prob) in self.transition(&state, &action) {
                    let reward = self.reward(&state, &action, &next_state);
                    println!("    {:>2?} {:>3.1}%  {}", next_state, prob * 100.0, reward);
                }
            }
        }
    }
}
