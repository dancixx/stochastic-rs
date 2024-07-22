use ndarray::{Array1, Array2};

/// A trait for generating stochastic process samples.
///
/// This trait defines methods for generating single and parallel samples of stochastic processes.
///
/// # Examples
///
/// ```
/// struct MyProcess;
///
/// impl Generator for MyProcess {
///     fn sample(&self) -> Array1<f64> {
///         vec![0.0, 1.0, 2.0]
///     }
///
///     fn sample_par(&self) -> Array2<f64>> {
///         vec![self.sample(), self.sample()]
///     }
/// }
///
/// let process = MyProcess;
/// let single_sample = process.sample();
/// let parallel_samples = process.sample_par();
/// ```
pub trait Generator: Sync + Send {
  /// Generates a single sample of the stochastic process.
  ///
  /// # Returns
  ///
  /// A `Array1<f64>` representing a single sample of the stochastic process.
  fn sample(&self) -> Array1<f64>;
  /// Generates parallel samples of the stochastic process.
  ///
  /// # Returns
  ///
  /// A `Array2<f64>>` where each inner vector represents a sample of the stochastic process.
  fn sample_par(&self) -> Array2<f64>;
}
