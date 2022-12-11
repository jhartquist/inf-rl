use crate::{
    mdp::MDP,
    policy::{MDPPolicy, Policy},
};
use std::collections::HashMap;

pub fn evaluate_policy<M, P>(mdp: &M, policy: &P) -> (HashMap<M::State, f64>, usize)
where
    M: MDP,
    P: Policy<M>,
{
    let mut state_values_prev = HashMap::new();

    let gamma = 0.99;
    let theta = 1e-10;

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
                    // TODO: only add future rewards if not in terminal state
                    prob * (reward + gamma * state_values_prev.get(next_state).unwrap_or(&0.0))
                })
                .sum();
            state_values.insert(state, state_value);
        }

        num_iterations += 1;

        let max_diff = mdp
            .states()
            .iter()
            .map(|s| {
                let n1 = *state_values.get(s).unwrap_or(&0.0);
                let n2 = *state_values_prev.get(s).unwrap_or(&0.0);
                (n1 - n2).abs()
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        if max_diff < theta {
            return (state_values, num_iterations);
        } else {
            state_values_prev = state_values.clone();
        }
    }
}

pub fn solve<M: MDP>(mdp: &M) {
    println!("{}", mdp.render());

    mdp.print_transitions();
    println!();

    let states = mdp.states();
    let actions = mdp.actions();

    // matches the example from GRDL on p.81
    let random_actions: Vec<_> = vec![3, 2, 1, 0, 2, 0, 3, 0, 0, 1, 0, 0, 0, 3, 1, 0]
        .into_iter()
        .map(|i| actions[i])
        .collect();

    let state_actions = states
        .iter()
        .zip(random_actions)
        .map(|(s, a)| (s.clone(), a.clone()))
        .collect();

    let policy = MDPPolicy::new(state_actions);

    println!("{}", mdp.render_policy(&policy));

    let (state_values, num_iterations) = evaluate_policy(mdp, &policy);
    println!("num iterations: {}", num_iterations);

    for (i, state) in states.iter().enumerate() {
        println!("{:>2} - {:>.4}", i, state_values.get(state).unwrap());
    }
}
