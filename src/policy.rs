use std::collections::HashMap;

use crate::mdp::MDP;

pub trait Policy<S, A> {
    fn get_action(&self, state: &S) -> A;
}

pub struct MDPPolicy<M: MDP> {
    state_actions: HashMap<M::State, M::Action>,
}

impl<M: MDP> MDPPolicy<M> {
    pub fn new(state_actions: HashMap<M::State, M::Action>) -> Self {
        Self { state_actions }
    }
}

impl<M: MDP> Policy<M::State, M::Action> for MDPPolicy<M> {
    fn get_action(&self, state: &M::State) -> M::Action {
        self.state_actions[state].clone()
    }
}
