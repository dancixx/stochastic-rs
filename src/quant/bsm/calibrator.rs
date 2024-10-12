use nalgebra::DVector;

#[derive(Clone, Debug)]
pub struct BSMParams {}

impl From<BSMParams> for DVector<f64> {
  fn from(_params: BSMParams) -> Self {
    DVector::from_vec(vec![])
  }
}

impl From<DVector<f64>> for BSMParams {
  fn from(_params: DVector<f64>) -> Self {
    BSMParams {}
  }
}

pub struct BSMCalibrator {}
