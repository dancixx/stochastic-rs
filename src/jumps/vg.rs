use crate::noises::gn;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Gamma;

pub fn vg(mu: f64, sigma: f64, nu: f64, n: usize, x0: Option<f64>, t: Option<f64>) -> Vec<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;

  let shape = dt / nu;
  let scale = nu;

  let mut vg = Array1::<f64>::zeros(n);
  vg[0] = x0.unwrap_or(0.0);

  let gn = gn::gn(n - 1, t);
  let gammas = Array1::random(n - 1, Gamma::new(shape, scale).unwrap());

  for i in 1..n {
    vg[i] = vg[i - 1] + mu * gammas[i - 1] + sigma * gammas[i - 1].sqrt() * gn[i - 1];
  }

  vg.to_vec()
}
