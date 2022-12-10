use std::collections::HashMap;

use crate::environment::Environment;

pub trait Policy<E: Environment> {
    fn get_action(&self, state: &E::State) -> E::Action;
}

pub struct DeterministicPolicy<E: Environment> {
    state_actions: HashMap<E::State, E::Action>,
}

impl<E: Environment> DeterministicPolicy<E> {
    pub fn new(state_actions: HashMap<E::State, E::Action>) -> Self {
        Self { state_actions }
    }
}

impl<E: Environment> Policy<E> for DeterministicPolicy<E> {
    fn get_action(&self, state: &E::State) -> E::Action {
        self.state_actions
            .get(state)
            .expect("state_actions includes state")
            .clone()
    }
}
