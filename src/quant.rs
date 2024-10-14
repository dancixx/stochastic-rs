use std::fmt::Display;

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

/// Moneyness.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Moneyness {
  #[default]
  DeepInTheMoney,
  InTheMoney,
  AtTheMoney,
  OutOfTheMoney,
  DeepOutOfTheMoney,
}

impl Display for Moneyness {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Moneyness::DeepInTheMoney => write!(f, "Deep in the money"),
      Moneyness::InTheMoney => write!(f, "In the money"),
      Moneyness::AtTheMoney => write!(f, "At the money"),
      Moneyness::OutOfTheMoney => write!(f, "Out of the money"),
      Moneyness::DeepOutOfTheMoney => write!(f, "Deep out of the money"),
    }
  }
}
