use ndarray::{Array0, Array1, Axis, Dim};
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

pub struct Poisson {
  pub lambda: f64,
  pub n: Option<usize>,
  pub t_max: Option<f64>,
}

pub fn poisson(params: &Poisson) -> Array1<f64> {
  let Poisson { lambda, n, t_max } = *params;
  if let Some(n) = n {
    let exponentials = Array1::random(n, Exp::new(1.0 / lambda).unwrap());
    let mut poisson = Array1::<f64>::zeros(n + 1);
    for i in 1..(n + 1) {
      poisson[i] = poisson[i - 1] + exponentials[i - 1];
    }

    poisson
  } else if let Some(t_max) = t_max {
    let mut poisson = Array1::from(vec![0.0]);
    let mut t = 0.0;

    while t < t_max {
      t += Exp::new(1.0 / lambda).unwrap().sample(&mut thread_rng());
      poisson
        .push(Axis(0), Array0::from_elem(Dim(()), t).view())
        .unwrap();
    }

    poisson
  } else {
    panic!("n or t_max must be provided");
  }
}
