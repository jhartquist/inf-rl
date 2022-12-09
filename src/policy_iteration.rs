use crate::{environment::Environment, mdp::FiniteMDP};
use std::collections::HashMap;

pub trait Policy<E: Environment> {
    fn get_action(&self, state: &E::State) -> E::Action;
}

struct DeterministicPolicy<E: Environment<State = usize>> {
    state_actions: Vec<E::Action>,
}

impl<E: Environment<State = usize>> Policy<E> for DeterministicPolicy<E> {
    fn get_action(&self, state: &usize) -> E::Action {
        self.state_actions[*state]
    }
}

pub fn evaluate_policy<M, P>(mdp: &M, policy: &P) -> HashMap<M::State, f64>
where
    M: FiniteMDP,
    P: Policy<M>,
{
    let mut state_values = HashMap::new();

    for state in mdp.states() {
        let action = policy.get_action(&state);
        let transitions = mdp.transition(&state, &action);

        let state_value = transitions
            .iter()
            .map(|(next_state, prob)| {
                let reward = mdp.reward(&state, &action, &next_state);
                // TODO: only add future rewards if not in terminal state
                prob * (reward + state_values.get(next_state).unwrap_or(&0.0))
            })
            .sum();

        state_values.insert(state, state_value);
    }

    state_values
}

pub fn solve<M: FiniteMDP<State = usize>>(mdp: &M) {
    let states = mdp.states();
    let actions = mdp.actions();

    let action = actions.last().expect("At least one action");

    let policy = DeterministicPolicy {
        state_actions: vec![action.clone(); states.len()],
    };

    println!("{:?}", policy.state_actions);

    // TODO: remove the type annotation by combining into only one generic type?
    let state_values = evaluate_policy::<M, DeterministicPolicy<M>>(&mdp, &policy);
    println!("state vals: {:?}", state_values);
}
