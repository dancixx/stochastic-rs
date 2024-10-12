pub mod bsm;
pub mod heston;
pub mod strategies;
pub mod r#trait;
#[cfg(feature = "yahoo")]
pub mod yahoo;

/// Option type.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OptionType {
  #[default]
  Call,
  Put,
}
