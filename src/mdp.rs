use std::hash::Hash;
use std::{cmp::Eq, collections::HashMap};

use itertools::{Itertools, Product};

use crate::environment::Reward;
use crate::policy::Policy;

#[allow(dead_code)]
type StateActionIter<'a, S, A> = Product<std::slice::Iter<'a, S>, std::slice::Iter<'a, A>>;

pub type Probability = f64;

pub trait MDP {
    type State: Copy + Hash + Eq;
    type Action: Copy + Hash + Eq;

    fn get_states(&self) -> &[Self::State];
    fn get_actions(&self) -> &[Self::Action];

    fn transition(&self, state: Self::State, action: Self::Action)
        -> &[(Self::State, Probability)];
    fn reward(&self, state: Self::State, action: Self::Action, next_state: Self::State) -> Reward;

    fn state_actions(&self) -> StateActionIter<'_, Self::State, Self::Action> {
        let states = self.get_states().into_iter();
        let actions = self.get_actions().into_iter();
        states.cartesian_product(actions)
    }

    fn render_policy<P>(&self, _policy: &P) -> String
    where
        P: Policy<Self::State, Self::Action>,
    {
        unimplemented!();
    }
}

//     fn print_transitions(&self) {
//         let actions = self.actions();
//         for state in self.states() {
//             println!("state {:>2?}", state);
//             for action in &actions {
//                 println!("  {}", action);
//                 for (next_state, prob) in self.transition(&state, &action) {
//                     let reward = self.reward(&state, &action, &next_state);
//                     println!("    {:>2?} {:>3.1}%  {}", next_state, prob * 100.0, reward);
//                 }
//             }
//         }
//     }

pub struct BasicMDP<S, A>
where
    S: Copy + Hash + Eq,
    A: Copy + Hash + Eq,
{
    states: Vec<S>,
    actions: Vec<A>,
    transitions: HashMap<(S, A), Vec<(S, Probability)>>,
    rewards: HashMap<(S, A, S), Reward>,
}

impl<S, A> BasicMDP<S, A>
where
    S: Copy + Hash + Eq,
    A: Copy + Hash + Eq,
{
    pub fn new(
        states: Vec<S>,
        actions: Vec<A>,
        transitions: HashMap<(S, A), Vec<(S, Probability)>>,
        rewards: HashMap<(S, A, S), Reward>,
    ) -> Self {
        BasicMDP {
            states,
            actions,
            transitions,
            rewards,
        }
    }
}

impl<S, A> MDP for BasicMDP<S, A>
where
    S: Copy + Hash + Eq,
    A: Copy + Hash + Eq,
{
    type State = S;
    type Action = A;

    fn get_states(&self) -> &[Self::State] {
        &self.states
    }

    fn get_actions(&self) -> &[Self::Action] {
        &self.actions
    }

    fn transition(&self, state: Self::State, action: Self::Action) -> &[(S, Probability)] {
        &self.transitions[&(state, action)]
    }

    fn reward(&self, state: Self::State, action: Self::Action, next_state: Self::State) -> Reward {
        self.rewards[&(state, action, next_state)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct VecMDP {
        states: Vec<u8>,
        actions: Vec<u16>,
        transitions: Vec<(u8, Reward)>,
    }

    impl MDP for VecMDP {
        type State = u8;
        type Action = u16;

        fn get_states(&self) -> &[Self::State] {
            &self.states
        }
        fn get_actions(&self) -> &[Self::Action] {
            &self.actions
        }

        fn transition(
            &self,
            _state: Self::State,
            _action: Self::Action,
        ) -> &[(Self::State, Probability)] {
            &self.transitions
        }

        fn reward(
            &self,
            _state: Self::State,
            _action: Self::Action,
            _next_state: Self::State,
        ) -> Reward {
            0.0
        }
    }

    #[test]
    fn mdp_with_vecs() {
        let mdp = VecMDP {
            states: vec![1, 2, 3],
            actions: vec![4, 5],
            transitions: vec![],
        };

        assert_eq!(mdp.get_states(), [1, 2, 3]);
        assert_eq!(mdp.get_actions(), [4, 5]);
        assert_eq!(
            mdp.state_actions()
                .map(|(&s, &a)| (s, a))
                .collect::<Vec<_>>(),
            vec![(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5),]
        );
    }
}
