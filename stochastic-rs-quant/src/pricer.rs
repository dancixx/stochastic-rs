use crate::ValueOrVec;

pub(crate) trait Pricer {
  /// Prices.
  fn prices(&self) -> Option<ValueOrVec<(f64, f64)>>;
  /// Derivatives.
  fn derivates(&self) -> Option<ValueOrVec<f64>>;
}
