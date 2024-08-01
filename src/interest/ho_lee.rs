use ndarray::Array1;

use crate::noises::gn;

#[allow(non_snake_case)]
pub struct HoLee<'a>
where
  'a: 'static,
{
  pub f_T: Option<Box<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,
  pub theta: Option<f64>,
  pub sigma: f64,
  pub n: usize,
  pub t: f64,
}

pub fn ho_lee(params: &HoLee) -> Array1<f64> {
  let HoLee {
    f_T,
    theta,
    sigma,
    n,
    t,
  } = params;
  assert!(
    theta.is_none() && f_T.is_none(),
    "theta or f_T must be provided"
  );
  let dt = *t / *n as f64;
  let gn = gn::gn(n - 1, Some(*t));

  let mut r = Array1::<f64>::zeros(*n);

  for i in 1..*n {
    let drift = if let Some(r#fn) = f_T {
      (r#fn)(i as f64 * dt) + sigma.powf(2.0)
    } else {
      theta.unwrap() + sigma.powf(2.0)
    };

    r[i] = r[i - 1] + drift * dt + sigma * gn[i - 1];
  }

  r
}
