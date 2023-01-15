use rand::{rngs::ThreadRng, seq::SliceRandom};

use crate::{
    mdp::MDP,
    policy::{DiscretePolicy, Policy},
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
    P: Policy<M::State, M::Action>,
{
    let mut state_values_prev: HashMap<M::State, f64> = mdp
        .get_states()
        .into_iter()
        .map(|s| (s.clone(), 0.0))
        .collect();

    let mut state_values = state_values_prev.clone();

    let mut num_iterations = 0;

    loop {
        for &state in mdp.get_states() {
            let action = policy.get_action(&state);
            let transitions = mdp.transition(state, action);
            let state_value = transitions
                .iter()
                .map(|&(next_state, prob)| {
                    let reward = mdp.reward(state, action, next_state);
                    let next_state_value = state_values_prev.get(&next_state).unwrap_or(&0.0);
                    prob * (reward + discount_rate * next_state_value)
                })
                .sum();
            state_values.insert(state.clone(), state_value);
        }

        num_iterations += 1;

        let max_diff = mdp
            .get_states()
            .iter()
            .map(|s| (state_values[s] - state_values_prev[s]).abs())
            .fold(f64::NEG_INFINITY, f64::max);

        if max_diff < threshold {
            println!("  {}", num_iterations);
            return (state_values, num_iterations);
        } else {
            (state_values_prev, state_values) = (state_values, state_values_prev);
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

    for (&state, &action) in mdp.state_actions() {
        let action_values = &mut state_action_values.entry(state).or_insert(HashMap::new());
        for &(next_state, prob) in mdp.transition(state, action) {
            let reward = mdp.reward(state, action, next_state);
            let action_value = action_values.entry(action).or_insert(0.0);
            let next_value = state_values.get(&next_state).unwrap_or(&0.0);
            *action_value += prob * (reward + discount_rate * next_value);
        }
    }

    let actions = mdp.get_actions();
    state_action_values
        .into_iter()
        .map(|(state, action_values)| {
            let best_action = action_values
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((*actions.first().unwrap(), 0.0))
                .0;
            (state.clone(), best_action.clone())
        })
        .collect()
}

pub fn policy_iteration<M>(
    mdp: &M,
    discount_rate: f64,
    threshold: f64,
    rng: &mut ThreadRng,
) -> DiscretePolicy<M::State, M::Action>
where
    M: MDP,
{
    let actions = mdp.get_actions();
    let states = mdp.get_states();

    // random policy
    let mut state_actions: HashMap<M::State, M::Action> = states
        .into_iter()
        .map(|state| {
            let action = actions.choose(rng).expect("at least one action").clone();
            (state.clone(), action)
        })
        .collect();

    let mut num_iterations = 0;
    loop {
        num_iterations += 1;

        let policy = DiscretePolicy::new(state_actions.clone());

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

pub fn value_iteration<M>(
    mdp: &M,
    discount_rate: f64,
    threshold: f64,
) -> DiscretePolicy<M::State, M::Action>
where
    M: MDP,
{
    let actions = mdp.get_actions();

    let mut state_values_prev: HashMap<M::State, f64> = mdp
        .get_states()
        .into_iter()
        .map(|s| (s.clone(), 0.0))
        .collect();

    let mut num_iterations = 0;

    let mut state_action_values: HashMap<M::State, HashMap<M::Action, f64>>;

    loop {
        num_iterations += 1;
        state_action_values = HashMap::new();

        for (&state, &action) in mdp.state_actions() {
            let action_values = &mut state_action_values
                .entry(state.clone())
                .or_insert(HashMap::new());
            for &(next_state, prob) in mdp.transition(state, action) {
                let reward = mdp.reward(state, action, next_state);
                let action_value = action_values.entry(action.clone()).or_insert(0.0);
                let next_value = state_values_prev.get(&next_state).unwrap_or(&0.0);
                *action_value += prob * (reward + discount_rate * next_value);
            }
        }

        let state_values: HashMap<M::State, f64> = state_action_values
            .iter()
            .map(|(state, action_values)| {
                let max_value = if action_values.len() > 0 {
                    action_values
                        .values()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                } else {
                    0.0
                };
                (state.clone(), max_value)
            })
            .collect();

        let max_diff = mdp
            .get_states()
            .iter()
            .map(|s| (state_values[s] - state_values_prev[s]).abs())
            .fold(f64::NEG_INFINITY, f64::max);

        if max_diff < threshold {
            println!("num_iterations: {}", num_iterations);
            break;
        } else {
            state_values_prev = state_values;
        }
    }

    let state_actions: HashMap<M::State, M::Action> = state_action_values
        .iter()
        .map(|(state, action_values)| {
            let best_action = action_values
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((actions.first().unwrap(), &0.0))
                .0;
            (state.clone(), best_action.clone())
        })
        .collect();

    DiscretePolicy::new(state_actions)
}
