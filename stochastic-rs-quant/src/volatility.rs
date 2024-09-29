pub mod heston;

use std::cell::RefCell;

use either::Either;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn, Owned};

/// Pricer trait.
pub(crate) trait Pricer {
  /// Calculate the price of an option.
  fn calculate_price(&mut self) -> Either<(f64, f64), Vec<(f64, f64)>>;
  /// Update the parameters.
  fn update_params(&mut self, params: DVector<f64>);
  /// Prices.
  fn prices(&self) -> Option<Either<(f64, f64), Vec<(f64, f64)>>>;
  /// Derivatives.
  fn derivates(&self) -> Option<Either<Vec<f64>, Vec<Vec<f64>>>>;
}

/// A calibrator.
pub(crate) struct Calibrator<'a, P>
where
  P: Pricer,
{
  /// Params to calibrate.
  pub params: DVector<f64>,
  /// Option prices from the market.
  pub c: Option<DVector<f64>>,
  /// Pricer backend.
  pricer: &'a RefCell<P>,
}

impl<'a, P> Calibrator<'a, P>
where
  P: Pricer,
{
  #[must_use]
  pub(crate) fn new(params: DVector<f64>, c: Option<DVector<f64>>, pricer: &'a RefCell<P>) -> Self {
    Self { params, c, pricer }
  }
}

impl<'a, P> LeastSquaresProblem<f64, Dyn, Dyn> for Calibrator<'a, P>
where
  P: Pricer,
{
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.pricer.borrow_mut().update_params(params.clone());
    self.params.copy_from(params);
  }

  fn params(&self) -> DVector<f64> {
    self.params.clone()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    self.pricer.borrow_mut().calculate_price();
    let options = self.pricer.borrow().prices().unwrap();
    let calls = options
      .as_ref()
      .right()
      .unwrap()
      .iter()
      .map(|x| x.0)
      .collect::<Vec<f64>>();

    let residuals = calls
      .iter()
      .zip(self.c.as_ref().unwrap().iter())
      .map(|(x, y)| x - y)
      .collect::<Vec<f64>>();

    Some(DVector::from_vec(residuals))
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let derivates = self.pricer.borrow().derivates().unwrap();
    let derivates = derivates
      .as_ref()
      .right()
      .unwrap()
      .iter()
      .flatten()
      .cloned()
      .collect::<Vec<f64>>();

    // The Jacobian matrix is a matrix of partial derivatives
    // of the residuals with respect to the parameters.
    let jacobian = DMatrix::from_vec(derivates.len() / self.params.len(), 5, derivates);
    Some(jacobian)
  }
}
