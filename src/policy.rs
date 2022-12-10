use crate::environment::Environment;

pub trait Policy<E: Environment> {
    fn get_action(&self, state: &E::State) -> E::Action;
}

pub struct DeterministicPolicy<E: Environment<State = usize>> {
    state_actions: Vec<E::Action>,
}

impl<E: Environment<State = usize>> DeterministicPolicy<E> {
    pub fn new(state_actions: Vec<E::Action>) -> Self {
        Self { state_actions }
    }
}

impl<E: Environment<State = usize>> Policy<E> for DeterministicPolicy<E> {
    fn get_action(&self, state: &usize) -> E::Action {
        self.state_actions[*state]
    }
}
