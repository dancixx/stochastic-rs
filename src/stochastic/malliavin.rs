use ndarray::Array1;

use super::Sampling;

pub trait Malliavin {
  /// Compute the Malliavin derivative of the stochastic process using perturbation.
  fn malliavin_derivate<F>(&self, f: F, epsilon: f64) -> Array1<f64>
  where
    F: Fn(&Array1<f64>) -> f64,
  {
    let path = self.path();
    let mut derivates = Array1::zeros(path.len() + 1);

    let f_original = f(&path);

    for i in 1..self.path().len() {
      let mut perturbed_path = path.clone();
      perturbed_path[i] += epsilon;

      let f_perturbed = f(&perturbed_path);

      derivates[i] = (f_perturbed - f_original) / epsilon;
    }

    derivates
  }

  /// Compute the Malliavin derivative of the stochastic process using perturbation for latest value.
  /// For example we want to know how the option price changes if the stock price changes.
  fn malliavin_derivate_latest(&self, epsilon: f64) -> Array1<f64> {
    let path = self.path();
    let mut derivates = Array1::zeros(path.len() + 1);

    let final_value = |path: &Array1<f64>| -> f64 { *path.last().unwrap() };
    let f_original = final_value(&path);

    for i in 1..path.len() {
      let mut perturbed_path = path.clone();
      perturbed_path[i] += epsilon; // Kis perturbáció a folyamat adott pontján

      let f_perturbed = final_value(&perturbed_path);

      derivates[i] = (f_perturbed - f_original) / epsilon;
    }

    derivates
  }

  /// Get stochastic process path.
  fn path(&self) -> Array1<f64>;
}

impl<T: Sampling<f64>> Malliavin for T {
  fn path(&self) -> Array1<f64> {
    self.sample()
  }
}
