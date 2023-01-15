pub trait DecaySchedule {
    fn get(&self, step: usize) -> f64;
}

pub struct LinearDecay {
    start: f64,
    end: f64,
    decay_steps: usize,
    delta: f64,
}

impl LinearDecay {
    pub fn new(start: f64, end: f64, decay_steps: usize) -> Self {
        let delta = (end - start) / decay_steps as f64;
        LinearDecay {
            start,
            end,
            decay_steps,
            delta,
        }
    }
}

impl DecaySchedule for LinearDecay {
    fn get(&self, step: usize) -> f64 {
        if step < self.decay_steps {
            self.start + self.delta * step as f64
        } else {
            self.end
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_decay() {
        let decay = LinearDecay::new(1.0, 0.1, 10);
        let values: Vec<_> = (0..15).map(|s| decay.get(s)).collect();
        assert_eq!(values[0], 1.0);
        assert_eq!(values[10], 0.1);
        assert_eq!(values[14], 0.1);
        assert!(values[1] < values[0]);
        assert!(values[9] > values[10]);
    }
}
