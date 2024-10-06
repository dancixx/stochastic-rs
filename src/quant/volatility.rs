pub mod heston;

use std::cell::RefCell;

use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn, Owned};

use super::{r#trait::Pricer, OptionType};

/// A calibrator.
pub(crate) struct Calibrator<'a, P>
where
  P: Pricer,
{
  /// Params to calibrate.
  pub params: DVector<f64>,
  /// Option prices from the market.
  pub c_market: DVector<f64>,
  /// Strike price vector.
  pub k: DVector<f64>,
  /// Option type
  pub option_type: &'a OptionType,
  /// Pricer backend.
  pricer: &'a RefCell<P>,
  /// Derivate matrix.
  derivates: RefCell<Vec<Vec<f64>>>,
}

impl<'a, P> Calibrator<'a, P>
where
  P: Pricer,
{
  #[must_use]
  pub(crate) fn new(
    params: DVector<f64>,
    c_market: Vec<f64>,
    k: Vec<f64>,
    option_type: &'a OptionType,
    pricer: &'a RefCell<P>,
  ) -> Self {
    Self {
      params,
      c_market: DVector::from_vec(c_market),
      k: DVector::from_vec(k),
      option_type,
      pricer,
      derivates: RefCell::new(Vec::new()),
    }
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
    let mut c_model = DVector::zeros(self.c_market.len());
    let mut derivates = Vec::new();

    for i in 0..c_model.len() {
      self.pricer.borrow_mut().update_strike(self.k[i]);
      self.pricer.borrow_mut().calculate_price();
      let (c, p) = self.pricer.borrow().prices();

      if self.option_type == &OptionType::Call {
        c_model[i] = c;
      } else {
        c_model[i] = p;
      }

      derivates.push(self.pricer.borrow().derivates());
    }

    self.derivates.replace(derivates);
    Some(c_model - self.c_market.clone())
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let derivates = self.derivates.borrow();
    let derivates = derivates.iter().flatten().cloned().collect::<Vec<f64>>();

    // The Jacobian matrix is a matrix of partial derivatives
    // of the residuals with respect to the parameters.
    let jacobian = DMatrix::from_vec(
      derivates.len() / self.params.len(),
      self.params.len(),
      derivates,
    );
    Some(jacobian)
  }
}
