use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

use super::vasicek::{self, Vasicek};

#[derive(Default)]
pub struct Mvasicek {
  pub mu: Array1<f64>,
  pub sigma: Array1<f64>,
  pub theta: Option<f64>,
  pub n: usize,
  pub m: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn mvasicek(params: &Mvasicek) -> Array2<f64> {
  let Mvasicek {
    mu,
    sigma,
    theta,
    n,
    m,
    x0,
    t,
  } = params;

  let mut xs = Array2::<f64>::zeros((*m, *n));

  for i in 0..*m {
    let vasicek = Vasicek {
      mu: mu[i],
      sigma: sigma[i],
      theta: *theta,
      n: *n,
      x0: *x0,
      t: *t,
    };

    xs.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x| {
      x.assign(&vasicek::vasicek(&vasicek));
    })
  }

  xs
}
