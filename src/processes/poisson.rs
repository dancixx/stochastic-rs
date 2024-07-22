use ndarray::{Array0, Array1, Axis, Dim};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::{Distribution, Exp};
use ndarray_rand::RandomExt;
use rand::thread_rng;

/// Generates a Poisson process.
///
/// The Poisson process models the occurrence of events over time. It is commonly used in fields such as queuing theory, telecommunications, and finance.
///
/// # Parameters
///
/// - `lambda`: Rate parameter (average number of events per unit time).
/// - `n`: Number of events (optional).
/// - `t_max`: Maximum time (optional).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated Poisson process path.
///
/// # Panics
///
/// Panics if neither `n` nor `t_max` is provided.
///
/// # Example
///
/// ```
/// let poisson_path = poisson(1.0, Some(1000), None);
/// let poisson_path = poisson(1.0, None, Some(100.0));
/// ```
pub fn poisson(lambda: f64, n: Option<usize>, t_max: Option<f64>) -> Array1<f64> {
  if let Some(n) = n {
    let exponentials = Array1::random(n - 1, Exp::new(1.0 / lambda).unwrap());
    let mut poisson = Array1::<f64>::zeros(n);

    for i in 1..n {
      poisson[i] = poisson[i - 1] + exponentials[i - 1];
    }

    poisson
  } else if let Some(t_max) = t_max {
    let mut poisson = Array1::from(vec![0.0]);
    let mut t = 0.0;

    while t < t_max {
      t += Exp::new(lambda).unwrap().sample(&mut thread_rng());
      poisson
        .push(Axis(0), Array0::from_elem(Dim(()), t).view())
        .unwrap();
    }

    poisson
  } else {
    panic!("n or t_max must be provided");
  }
}

/// Generates a compound Poisson process.
///
/// The compound Poisson process models the occurrence of events over time, where each event has a random magnitude (jump). It is commonly used in insurance and finance.
///
/// # Parameters
///
/// - `n`: Number of time steps.
/// - `lambda`: Rate parameter (average number of events per unit time).
/// - `jumps`: Vector of jump sizes (optional).
/// - `t_max`: Maximum time (optional, defaults to 1.0).
/// - `jump_mean`: Mean of the jump sizes (optional, defaults to 0.0).
/// - `jump_std`: Standard deviation of the jump sizes (optional, defaults to 1.0).
///
/// # Returns
///
/// A `(Array1<f64>, Array1<f64>, Array1<f64>)` representing the exponetial times from Poisson, generated compound Poisson cumulative process path and the jumps.
///
/// # Panics
///
/// Panics if `n` is zero.
///
/// # Example
///
/// ```
/// let (p, cum_cp, cp) = compound_poisson(1000, 2.0, None, Some(10.0), Some(0.0), Some(1.0));
/// ```
pub fn compound_poisson(
  n: usize,
  lambda: f64,
  t_max: Option<f64>,
  jump_mean: Option<f64>,
  jump_std: Option<f64>,
) -> [Array1<f64>; 3] {
  if n == 0 {
    panic!("n must be a positive integer");
  }

  let _t_max = t_max.unwrap_or(1.0);
  let p = poisson(lambda, Some(n), t_max);
  let mut cp = Array1::<f64>::zeros(n);

  let _jump_mean = jump_mean.unwrap_or(0.0);
  let _jump_std = jump_std.unwrap_or(1.0);

  for i in 1..p.len() {
    let norm = Array1::random(i, Normal::new(_jump_mean, _jump_std).unwrap());
    cp[i] = norm.sum();
  }

  let mut cum_cp = cp.clone();
  cum_cp.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

  [p, cum_cp, cp]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_poisson() {
    let n = 1000;
    let lambda = 1;
    let p = poisson(lambda as f64, Some(n), None);
    println!("{:?}", p.len());
    let t = 100.0;
    let p = poisson(lambda as f64, None, Some(t));
    println!("{:?}", p.len());
  }

  #[test]
  fn test_compound_poisson() {
    let n = 1000;
    let lambda = 2.0;
    let t = 10.0;
    let [.., cp] = compound_poisson(n, lambda, None, Some(t), None);
    assert_eq!(cp.len(), n);
  }
}
