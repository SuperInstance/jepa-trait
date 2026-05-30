//! JEPA (Joint Embedding Predictive Architecture) as a pluggable trait.
//!
//! In the Grand Pattern, JEPA is a READING — it learns to weight prior readings
//! specific to a room. The weighted history IS the room's voice. JEPA formulates
//! the room's state as prompt context for agents entering that room.

/// A single reading from a room's vibe history.
#[derive(Debug, Clone)]
pub struct Reading {
    /// Timestamp (monotonic tick count)
    pub tick: u64,
    /// The mono-dimensional vibe value
    pub vibe: f64,
    /// Confidence weight for this reading
    pub confidence: f64,
}

/// A JEPA prediction: what the room's vibe will be next tick.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted vibe value
    pub predicted_vibe: f64,
    /// Prediction confidence [0, 1]
    pub confidence: f64,
    /// Which past readings were weighted most (indices into history)
    pub weighted_indices: Vec<(usize, f64)>,
}

/// The JEPA trait: a pluggable predictor that learns to weight prior readings.
///
/// Each room/venue has its own JEPA instance. The JEPA reads the room's weighted
/// history and formulates it as context. "Here's what this room is feeling."
pub trait JepaPredictor: Send + Sync {
    /// Create a new predictor with given history capacity.
    fn new(capacity: usize) -> Self where Self: Sized;

    /// Record a new reading.
    fn observe(&mut self, reading: Reading);

    /// Predict the next vibe value.
    fn predict(&self) -> Prediction;

    /// Get the current weighted history.
    fn history(&self) -> &[Reading];

    /// Get the room's voice: a formatted string of the room's current state.
    /// This is what gets prompt-injected into agents entering the room.
    fn room_voice(&self) -> String {
        let pred = self.predict();
        let history = self.history();
        let recent_avg = if history.is_empty() {
            0.0
        } else {
            let n = history.len().min(5);
            history[history.len() - n..].iter().map(|r| r.vibe).sum::<f64>() / n as f64
        };
        format!(
            "Room vibe: {:.3} (trend: {:.3}, confidence: {:.1}%). Recent average: {:.3}. History depth: {} readings.",
            pred.predicted_vibe,
            pred.predicted_vibe - recent_avg,
            pred.confidence * 100.0,
            recent_avg,
            history.len()
        )
    }

    /// Number of readings stored.
    fn depth(&self) -> usize {
        self.history().len()
    }

    /// Clear all history.
    fn reset(&mut self);
}

/// Exponential decay JEPA: weights recent readings exponentially more.
pub struct ExponentialJepa {
    history: Vec<Reading>,
    capacity: usize,
    decay: f64, // exponential decay factor, e.g. 0.9
}

impl ExponentialJepa {
    pub fn with_decay(capacity: usize, decay: f64) -> Self {
        Self { history: Vec::new(), capacity, decay }
    }

    fn compute_weights(&self) -> Vec<f64> {
        let n = self.history.len();
        self.history.iter().enumerate().map(|(i, _)| {
            let age = (n - 1 - i) as f64;
            self.decay.powf(age)
        }).collect()
    }
}

impl JepaPredictor for ExponentialJepa {
    fn new(capacity: usize) -> Self {
        Self { history: Vec::new(), capacity, decay: 0.9 }
    }

    fn observe(&mut self, reading: Reading) {
        if self.history.len() >= self.capacity {
            self.history.remove(0);
        }
        self.history.push(reading);
    }

    fn predict(&self) -> Prediction {
        if self.history.is_empty() {
            return Prediction { predicted_vibe: 0.0, confidence: 0.0, weighted_indices: vec![] };
        }
        let weights = self.compute_weights();
        let total_w: f64 = weights.iter().sum();
        let weighted_vibe: f64 = weights.iter().zip(self.history.iter())
            .map(|(w, r)| w * r.vibe * r.confidence)
            .sum();
        let predicted = weighted_vibe / total_w;
        let avg_confidence: f64 = self.history.iter().map(|r| r.confidence).sum::<f64>() / self.history.len() as f64;
        let weighted_indices: Vec<(usize, f64)> = weights.iter().enumerate()
            .filter(|(_, w)| **w > 0.01)
            .map(|(i, w)| (i, *w))
            .collect();
        Prediction { predicted_vibe: predicted, confidence: avg_confidence, weighted_indices }
    }

    fn history(&self) -> &[Reading] { &self.history }

    fn reset(&mut self) { self.history.clear(); }
}

/// Moving average JEPA: simple uniform weighting over a window.
pub struct MovingAverageJepa {
    history: Vec<Reading>,
    capacity: usize,
    window: usize,
}

impl MovingAverageJepa {
    pub fn with_window(capacity: usize, window: usize) -> Self {
        Self { history: Vec::new(), capacity, window }
    }
}

impl JepaPredictor for MovingAverageJepa {
    fn new(capacity: usize) -> Self {
        Self { history: Vec::new(), capacity, window: 5 }
    }

    fn observe(&mut self, reading: Reading) {
        if self.history.len() >= self.capacity {
            self.history.remove(0);
        }
        self.history.push(reading);
    }

    fn predict(&self) -> Prediction {
        if self.history.is_empty() {
            return Prediction { predicted_vibe: 0.0, confidence: 0.0, weighted_indices: vec![] };
        }
        let n = self.history.len().min(self.window);
        let window = &self.history[self.history.len() - n..];
        let predicted: f64 = window.iter().map(|r| r.vibe * r.confidence).sum::<f64>()
            / window.iter().map(|r| r.confidence).sum::<f64>().max(f64::EPSILON);
        let avg_conf: f64 = window.iter().map(|r| r.confidence).sum::<f64>() / n as f64;
        let indices: Vec<(usize, f64)> = (self.history.len() - n..self.history.len())
            .map(|i| (i, 1.0 / n as f64))
            .collect();
        Prediction { predicted_vibe: predicted, confidence: avg_conf, weighted_indices: indices }
    }

    fn history(&self) -> &[Reading] { &self.history }

    fn reset(&mut self) { self.history.clear(); }
}

/// Trend JEPA: predicts based on linear trend of recent readings.
pub struct TrendJepa {
    history: Vec<Reading>,
    capacity: usize,
}

impl JepaPredictor for TrendJepa {
    fn new(capacity: usize) -> Self {
        Self { history: Vec::new(), capacity }
    }

    fn observe(&mut self, reading: Reading) {
        if self.history.len() >= self.capacity {
            self.history.remove(0);
        }
        self.history.push(reading);
    }

    fn predict(&self) -> Prediction {
        if self.history.is_empty() {
            return Prediction { predicted_vibe: 0.0, confidence: 0.0, weighted_indices: vec![] };
        }
        if self.history.len() < 2 {
            let r = &self.history[0];
            return Prediction {
                predicted_vibe: r.vibe,
                confidence: r.confidence * 0.5,
                weighted_indices: vec![(0, 1.0)],
            };
        }
        // Simple linear regression: vibe = a * tick + b
        let n = self.history.len() as f64;
        let sum_x: f64 = self.history.iter().map(|r| r.tick as f64).sum();
        let sum_y: f64 = self.history.iter().map(|r| r.vibe).sum();
        let sum_xy: f64 = self.history.iter().map(|r| (r.tick as f64) * r.vibe).sum();
        let sum_x2: f64 = self.history.iter().map(|r| (r.tick as f64).powi(2)).sum();
        let denom = n * sum_x2 - sum_x * sum_x;
        let a = if denom.abs() < f64::EPSILON {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denom
        };
        let b = (sum_y - a * sum_x) / n;
        let last_tick = self.history.last().unwrap().tick;
        let predicted = a * (last_tick + 1) as f64 + b;
        // Confidence based on fit quality
        let residuals: f64 = self.history.iter()
            .map(|r| (r.vibe - (a * r.tick as f64 + b)).powi(2))
            .sum();
        let variance = if n > 1.0 { residuals / (n - 1.0) } else { residuals };
        let confidence = 1.0 / (1.0 + variance).min(1.0);
        let indices: Vec<(usize, f64)> = self.history.iter().enumerate()
            .rev().take(5)
            .map(|(i, _)| (i, 1.0))
            .collect();
        Prediction { predicted_vibe: predicted, confidence, weighted_indices: indices }
    }

    fn history(&self) -> &[Reading] { &self.history }

    fn reset(&mut self) { self.history.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reading(tick: u64, vibe: f64) -> Reading {
        Reading { tick, vibe, confidence: 1.0 }
    }

    #[test]
    fn test_exponential_empty() {
        let jepa = ExponentialJepa::new(10);
        let pred = jepa.predict();
        assert_eq!(pred.predicted_vibe, 0.0);
        assert_eq!(pred.confidence, 0.0);
    }

    #[test]
    fn test_exponential_single() {
        let mut jepa = ExponentialJepa::new(10);
        jepa.observe(make_reading(0, 0.5));
        let pred = jepa.predict();
        assert!((pred.predicted_vibe - 0.5).abs() < 1e-10);
        assert!(!pred.weighted_indices.is_empty());
    }

    #[test]
    fn test_exponential_recency_bias() {
        let mut jepa = ExponentialJepa::new(10);
        for i in 0..10 {
            jepa.observe(make_reading(i as u64, i as f64));
        }
        let pred = jepa.predict();
        // Should be biased toward recent high values
        assert!(pred.predicted_vibe > 5.0);
    }

    #[test]
    fn test_exponential_capacity() {
        let mut jepa = ExponentialJepa::new(3);
        for i in 0..10 {
            jepa.observe(make_reading(i as u64, i as f64));
        }
        assert_eq!(jepa.depth(), 3);
    }

    #[test]
    fn test_exponential_room_voice() {
        let mut jepa = ExponentialJepa::new(10);
        jepa.observe(make_reading(0, 0.8));
        let voice = jepa.room_voice();
        assert!(voice.contains("Room vibe"));
        assert!(voice.contains("0.8"));
    }

    #[test]
    fn test_moving_average_basic() {
        let mut jepa = MovingAverageJepa::new(10);
        for i in 0..5 {
            jepa.observe(make_reading(i as u64, 1.0));
        }
        let pred = jepa.predict();
        assert!((pred.predicted_vibe - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average_windowed() {
        let mut jepa = MovingAverageJepa::with_window(10, 3);
        for i in 0..10 {
            jepa.observe(make_reading(i as u64, i as f64));
        }
        let pred = jepa.predict();
        // Window of 3: should average 7, 8, 9
        assert!((pred.predicted_vibe - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average_reset() {
        let mut jepa = MovingAverageJepa::new(10);
        jepa.observe(make_reading(0, 1.0));
        jepa.reset();
        assert_eq!(jepa.depth(), 0);
    }

    #[test]
    fn test_trend_upward() {
        let mut jepa = TrendJepa::new(10);
        for i in 0..5 {
            jepa.observe(make_reading(i as u64, i as f64));
        }
        let pred = jepa.predict();
        assert!(pred.predicted_vibe > 4.0); // Should predict continuation upward
    }

    #[test]
    fn test_trend_flat() {
        let mut jepa = TrendJepa::new(10);
        for i in 0..10 {
            jepa.observe(make_reading(i as u64, 5.0));
        }
        let pred = jepa.predict();
        assert!((pred.predicted_vibe - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_trend_single_point() {
        let mut jepa = TrendJepa::new(10);
        jepa.observe(make_reading(0, 3.0));
        let pred = jepa.predict();
        assert!((pred.predicted_vibe - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_weighting() {
        let mut jepa = ExponentialJepa::new(10);
        jepa.observe(Reading { tick: 0, vibe: 1.0, confidence: 0.1 });
        jepa.observe(Reading { tick: 1, vibe: 0.0, confidence: 1.0 });
        let pred = jepa.predict();
        // High confidence reading should pull toward 0.0
        assert!(pred.predicted_vibe < 0.5);
    }

    #[test]
    fn test_venue_voice_format() {
        let mut jepa = ExponentialJepa::new(10);
        jepa.observe(make_reading(0, 0.75));
        jepa.observe(make_reading(1, 0.80));
        let voice = jepa.room_voice();
        assert!(voice.contains("confidence"));
        assert!(voice.contains("depth"));
    }

    #[test]
    fn test_weighted_indices_nonempty() {
        let mut jepa = ExponentialJepa::new(10);
        for i in 0..5 {
            jepa.observe(make_reading(i as u64, 0.5));
        }
        let pred = jepa.predict();
        assert!(!pred.weighted_indices.is_empty());
        // Most recent should have highest weight
        let max_weight_idx = pred.weighted_indices.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        assert_eq!(max_weight_idx.0, 4); // Last observation
    }

    #[test]
    fn test_dynamic_dispatch() {
        let predictors: Vec<Box<dyn JepaPredictor>> = vec![
            Box::new(ExponentialJepa::new(10)),
            Box::new(MovingAverageJepa::new(10)),
            Box::new(TrendJepa::new(10)),
        ];
        assert_eq!(predictors.len(), 3);
    }

    #[test]
    fn test_multiple_predictors_agree_on_constant() {
        let mut exp = ExponentialJepa::new(10);
        let mut ma = MovingAverageJepa::new(10);
        let mut trend = TrendJepa::new(10);
        for i in 0..20 {
            let r = make_reading(i as u64, 3.0);
            exp.observe(r.clone());
            ma.observe(r.clone());
            trend.observe(r.clone());
        }
        let exp_pred = exp.predict().predicted_vibe;
        let ma_pred = ma.predict().predicted_vibe;
        let trend_pred = trend.predict().predicted_vibe;
        assert!((exp_pred - 3.0).abs() < 0.1);
        assert!((ma_pred - 3.0).abs() < 0.1);
        assert!((trend_pred - 3.0).abs() < 0.01);
    }
}
