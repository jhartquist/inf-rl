use std::collections::HashMap;
use std::hash::Hash;

pub trait Policy<S, A> {
    fn get_action(&self, state: &S) -> A;
}

pub struct DiscretePolicy<S, A>
where
    S: Hash + Eq,
    A: Clone,
{
    state_actions: HashMap<S, A>,
}

impl<S, A> DiscretePolicy<S, A>
where
    S: Hash + Eq,
    A: Clone,
{
    pub fn new(state_actions: HashMap<S, A>) -> Self {
        DiscretePolicy { state_actions }
    }
}

impl<S, A> Policy<S, A> for DiscretePolicy<S, A>
where
    S: Hash + Eq,
    A: Clone,
{
    fn get_action(&self, state: &S) -> A {
        self.state_actions[state].clone()
    }
}
