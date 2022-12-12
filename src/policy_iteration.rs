use rand::seq::SliceRandom;

use crate::{
    mdp::MDP,
    policy::{MDPPolicy, Policy},
};
use std::collections::HashMap;

fn evaluate_policy<M, P>(
    mdp: &M,
    policy: &P,
    discount_rate: f64,
    threshold: f64,
) -> (HashMap<M::State, f64>, usize)
where
    M: MDP,
    P: Policy<M>,
{
    let mut state_values_prev = HashMap::new();

    let mut num_iterations = 0;

    loop {
        let mut state_values = HashMap::new();

        for state in mdp.states() {
            let action = policy.get_action(&state);
            let transitions = mdp.transition(&state, &action);
            let state_value = transitions
                .iter()
                .map(|(next_state, prob)| {
                    let reward = mdp.reward(&state, &action, &next_state);
                    prob * (reward
                        + discount_rate * state_values_prev.get(next_state).unwrap_or(&0.0))
                })
                .sum();
            state_values.insert(state, state_value);
        }

        num_iterations += 1;

        let max_diff = mdp
            .states()
            .iter()
            .map(|s| {
                // TODO: refactor this out
                let n1 = *state_values.get(s).unwrap_or(&0.0);
                let n2 = *state_values_prev.get(s).unwrap_or(&0.0);
                (n1 - n2).abs()
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        if max_diff < threshold {
            return (state_values, num_iterations);
        } else {
            state_values_prev = state_values.clone();
        }
    }
}

fn improve_policy<M>(
    mdp: &M,
    state_values: HashMap<M::State, f64>,
    discount_rate: f64,
) -> HashMap<M::State, M::Action>
where
    M: MDP,
{
    let mut state_action_values = HashMap::new();

    let actions = mdp.actions();
    for state in mdp.states() {
        for action in &actions {
            let action_values = &mut state_action_values.entry(state).or_insert(HashMap::new());
            for (next_state, prob) in mdp.transition(&state, action) {
                let reward = mdp.reward(&state, action, &next_state);
                let action_value = action_values.entry(action).or_insert(0.0);
                *action_value +=
                    prob * (reward + discount_rate * state_values.get(&next_state).unwrap_or(&0.0));
            }
        }
    }

    state_action_values
        .into_iter()
        .map(|(state, action_values)| {
            let best_action = action_values
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((actions.first().unwrap(), 0.0))
                .0;
            (state, best_action.clone())
        })
        .collect()
}

pub fn policy_iteration<M>(mdp: &M, discount_rate: f64, threshold: f64) -> MDPPolicy<M>
where
    M: MDP,
{
    let actions = mdp.actions();
    let states = mdp.states();

    let mut rng = rand::thread_rng();

    let mut state_actions: HashMap<M::State, M::Action> = states
        .into_iter()
        .map(|state| {
            let action = actions
                .choose(&mut rng)
                .expect("at least one action")
                .clone();
            (state, action)
        })
        .collect();

    let mut num_iterations = 0;
    loop {
        num_iterations += 1;

        let policy = MDPPolicy::new(state_actions.clone());
        mdp.render_policy(&policy);

        let (state_values, _) = evaluate_policy(mdp, &policy, discount_rate, threshold);
        let new_state_actions = improve_policy(mdp, state_values, discount_rate);

        if state_actions == new_state_actions {
            println!("num iterations: {}", num_iterations);
            return policy;
        } else {
            state_actions = new_state_actions;
        }
    }
}
