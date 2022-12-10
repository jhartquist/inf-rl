use crate::{
    mdp::MPD,
    policy::{DeterministicPolicy, Policy},
};
use std::collections::HashMap;

pub fn evaluate_policy<M, P>(mdp: &M, policy: &P) -> HashMap<M::State, f64>
where
    M: MPD,
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

pub fn solve<M: MPD>(mdp: &M) {
    println!("{}", mdp.render());

    let states = mdp.states();
    let actions = mdp.actions();

    let action = actions.last().expect("At least one action");

    let state_actions = states.iter().map(|s| (s.clone(), action.clone())).collect();
    let policy = DeterministicPolicy::new(state_actions);

    println!("{}", mdp.render_policy(&policy));

    // TODO: remove the type annotation by combining into only one generic type?
    let state_values = evaluate_policy::<M, DeterministicPolicy<M>>(&mdp, &policy);

    for (i, state) in states.iter().enumerate() {
        println!("{:>2} - {:>.3}", i, state_values.get(state).unwrap());
    }
}
