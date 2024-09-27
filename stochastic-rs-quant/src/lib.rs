use std::mem::ManuallyDrop;

pub mod pricing;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// A value or a vector of values.
pub union ValueOrVec<T>
where
  T: Copy,
{
  pub x: T,
  pub v: ManuallyDrop<Vec<T>>,
}

/// Implement the `Clone` trait for `ValueOrVec<T>`.
impl Clone for ValueOrVec<f64> {
  fn clone(&self) -> Self {
    unsafe {
      Self {
        v: ManuallyDrop::new(self.v.clone().to_vec()),
      }
    }
  }
}

/// Implement the `Clone` trait for `ValueOrVec<T>`.
impl Clone for ValueOrVec<chrono::NaiveDate> {
  fn clone(&self) -> Self {
    unsafe {
      Self {
        v: ManuallyDrop::new(self.v.clone().to_vec()),
      }
    }
  }
}
