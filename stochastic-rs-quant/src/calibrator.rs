use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use crate::pricer::Pricer;

/// A calibrator.
pub(crate) struct Calibrator<'a, P>
where
  P: Pricer,
{
  pub p: DVector<f64>,
  pub market_prices: Option<DVector<f64>>,
  pricer: &'a P,
}

impl<'a, P> Calibrator<'a, P>
where
  P: Pricer,
{
  #[must_use]
  pub(crate) fn new(p: DVector<f64>, market_prices: Option<DVector<f64>>, pricer: &'a P) -> Self {
    Self {
      p,
      market_prices,
      pricer,
    }
  }
}

impl<'a, P> Calibrator<'a, P>
where
  P: Pricer,
{
  fn generate_market_prices(&self) -> DVector<f64> {
    let model_prices = self.pricer.prices().unwrap();
    let call_prices = unsafe {
      model_prices
        .v
        .clone()
        .iter()
        .map(|x| x.0)
        .collect::<Vec<f64>>()
    };

    // Add some noise to the market prices
    let market_prices = call_prices
      .iter()
      .map(|x| *x + Normal::new(20.0, 4.5).unwrap().sample(&mut thread_rng()))
      .collect::<Vec<f64>>();

    DVector::from_vec(market_prices)
  }
}

impl<'a, P> LeastSquaresProblem<f64, Dyn, Dyn> for Calibrator<'a, P>
where
  P: Pricer,
{
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, p: &DVector<f64>) {
    self.p.copy_from(p);
  }

  fn params(&self) -> DVector<f64> {
    self.p.clone()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let model_prices = self.pricer.prices().unwrap();
    let call_prices = unsafe {
      model_prices
        .v
        .clone()
        .iter()
        .map(|x| x.0)
        .collect::<Vec<f64>>()
    };

    let market_prices = match &self.market_prices {
      Some(x) => x.clone(),
      None => self.generate_market_prices(),
    };

    let residuals = call_prices
      .iter()
      .zip(market_prices.iter())
      .map(|(x, y)| x - y)
      .collect::<Vec<f64>>();

    Some(DVector::from_vec(residuals))
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let derivates = self.pricer.derivates().unwrap();
    let derivates = unsafe { derivates.v.clone().to_vec() };

    // The Jacobian matrix is a matrix of partial derivatives
    // of the residuals with respect to the parameters.
    let jacobian = DMatrix::from_vec(derivates.len() / self.p.len(), 5, derivates);
    Some(jacobian)
  }
}
