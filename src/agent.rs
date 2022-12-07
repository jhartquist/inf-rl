use crate::environment::Environment;

pub trait Agent<E: Environment> {
    fn act(&mut self, env: &mut E) -> E::Action;
}
